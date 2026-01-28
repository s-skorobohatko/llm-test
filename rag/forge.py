import os
import re
import time
import json
import tarfile
import sqlite3
import requests
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse


GIT_URL_RE = re.compile(r"^(https?|git)://|^git@")


def is_probable_git_repo(url: str) -> bool:
    if not url:
        return False
    u = url.strip()
    if not GIT_URL_RE.search(u):
        return False
    # Most Forge metadata.source is github/http(s). Good enough.
    return True


def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def git_sync(url: str, dest: str) -> None:
    safe_mkdir(os.path.dirname(dest))
    if os.path.isdir(dest) and os.path.isdir(os.path.join(dest, ".git")):
        subprocess.check_call(["git", "-C", dest, "pull", "--ff-only"])
    else:
        subprocess.check_call(["git", "clone", "--depth", "1", url, dest])


def ensure_forge_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS forge_state (
            module_slug TEXT PRIMARY KEY,
            last_version TEXT,
            last_kind TEXT,          -- "git" or "tar"
            last_checked INTEGER
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_forge_state_checked ON forge_state(last_checked);")


def get_forge_state(conn: sqlite3.Connection, slug: str) -> Optional[Tuple[str, str]]:
    row = conn.execute(
        "SELECT last_version, last_kind FROM forge_state WHERE module_slug=?",
        (slug,),
    ).fetchone()
    if not row:
        return None
    return (row[0], row[1])


def upsert_forge_state(conn: sqlite3.Connection, slug: str, version: str, kind: str) -> None:
    conn.execute(
        """
        INSERT INTO forge_state(module_slug, last_version, last_kind, last_checked)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(module_slug) DO UPDATE SET
          last_version=excluded.last_version,
          last_kind=excluded.last_kind,
          last_checked=excluded.last_checked
        """,
        (slug, version, kind, int(time.time())),
    )


def forge_list_modules(api_base: str, limit: int, offset: int) -> Dict[str, Any]:
    # Forge API v3: /v3/modules?limit=...&offset=...
    url = f"{api_base.rstrip('/')}/v3/modules"
    r = requests.get(url, params={"limit": limit, "offset": offset}, timeout=60)
    r.raise_for_status()
    return r.json()


def forge_get_module(api_base: str, slug: str) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/v3/modules/{slug}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def forge_download_file(api_base: str, file_uri: str, dest_path: str) -> None:
    """
    file_uri usually looks like: /v3/files/<filename>
    """
    safe_mkdir(os.path.dirname(dest_path))
    url = f"{api_base.rstrip('/')}{file_uri}"
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 512):
                if chunk:
                    f.write(chunk)


def safe_extract_tgz(tgz_path: str, dest_dir: str) -> None:
    safe_mkdir(dest_dir)

    def is_within_directory(directory: str, target: str) -> bool:
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

    with tarfile.open(tgz_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = os.path.join(dest_dir, member.name)
            if not is_within_directory(dest_dir, member_path):
                raise RuntimeError(f"Unsafe tar path traversal detected: {member.name}")
        tar.extractall(dest_dir)


def slug_to_paths(dest_root: str, slug: str) -> Tuple[str, str, str]:
    """
    Returns:
      owner, name, base_dir
    """
    owner, name = slug.split("-", 1) if "-" in slug else (slug, "unknown")
    # Forge slug format is often owner-name (dash), e.g. puppetlabs-stdlib
    # Keep both, and store under dest_root/owner/name
    base_dir = os.path.join(dest_root, owner, name)
    return owner, name, base_dir


def discover_and_sync_forge(
    conn: sqlite3.Connection,
    api_base: str,
    dest_root: str,
    include_globs: List[str],
    limit_per_page: int = 50,
    max_modules: int = 200,
    request_delay_sec: float = 0.2,
    only_with_repo: bool = True,
    only_owner: Optional[str] = None,
) -> List[str]:
    """
    Discovers Forge modules and syncs them locally (git clone/pull or tarball download).
    Returns list of local directories that should be indexed.
    """
    safe_mkdir(dest_root)
    ensure_forge_state_table(conn)

    synced_dirs: List[str] = []
    seen = 0
    offset = 0

    while seen < max_modules:
        page = forge_list_modules(api_base, limit_per_page, offset)
        results = page.get("results") or []
        if not results:
            break

        for item in results:
            if seen >= max_modules:
                break

            slug = item.get("slug")
            if not slug:
                continue

            # optional owner filter based on slug prefix
            if only_owner and not slug.startswith(f"{only_owner}-"):
                continue

            # Fetch full module detail (to get current_release.metadata.source + file_uri)
            mod = forge_get_module(api_base, slug)
            current = mod.get("current_release") or {}
            version = current.get("version") or ""
            metadata = current.get("metadata") or {}
            repo_url = (metadata.get("source") or "").strip()

            # Decide sync strategy
            use_git = is_probable_git_repo(repo_url)
            if only_with_repo and not use_git:
                continue

            prev = get_forge_state(conn, slug)
            if prev and prev[0] == version:
                # unchanged; still index existing directory if present
                _, _, base_dir = slug_to_paths(dest_root, slug)
                if use_git:
                    repo_dir = os.path.join(base_dir, "repo")
                    if os.path.isdir(repo_dir):
                        synced_dirs.append(repo_dir)
                else:
                    rel_dir = os.path.join(base_dir, "releases", version)
                    if os.path.isdir(rel_dir):
                        synced_dirs.append(rel_dir)
                seen += 1
                continue

            owner, name, base_dir = slug_to_paths(dest_root, slug)

            try:
                if use_git:
                    repo_dir = os.path.join(base_dir, "repo")
                    git_sync(repo_url, repo_dir)
                    upsert_forge_state(conn, slug, version, "git")
                    synced_dirs.append(repo_dir)
                else:
                    file_uri = current.get("file_uri")
                    if not file_uri:
                        continue
                    tgz_dir = os.path.join(base_dir, "downloads")
                    safe_mkdir(tgz_dir)
                    tgz_path = os.path.join(tgz_dir, f"{slug}-{version}.tar.gz")
                    forge_download_file(api_base, file_uri, tgz_path)
                    rel_dir = os.path.join(base_dir, "releases", version)
                    safe_extract_tgz(tgz_path, rel_dir)
                    upsert_forge_state(conn, slug, version, "tar")
                    synced_dirs.append(rel_dir)

                conn.commit()
                seen += 1

            except Exception as e:
                # keep going
                print(f"[forge] ERROR syncing {slug}: {e}")

            time.sleep(request_delay_sec)

        offset += limit_per_page

    # Deduplicate
    uniq = []
    seen_set = set()
    for d in synced_dirs:
        if d not in seen_set:
            uniq.append(d)
            seen_set.add(d)
    return uniq
