import os
import re
import time
import json
import tarfile
import requests
import subprocess
from typing import Dict, Any, List, Optional, Tuple


GIT_URL_RE = re.compile(r"^(https?|git)://|^git@")


def is_probable_git_repo(url: str) -> bool:
    if not url:
        return False
    return bool(GIT_URL_RE.search(url.strip()))


def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def atomic_write_json(path: str, obj: dict) -> None:
    safe_mkdir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_state(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {"modules": {}}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {"modules": {}}


def save_state(path: str, state: dict) -> None:
    atomic_write_json(path, state)


def get_state_entry(state: dict, slug: str) -> Optional[Tuple[str, str, int]]:
    ent = (state.get("modules") or {}).get(slug)
    if not ent:
        return None
    return (
        ent.get("last_version", "") or "",
        ent.get("last_kind", "") or "",
        int(ent.get("last_checked", 0) or 0),
    )


def upsert_state_entry(state: dict, slug: str, version: str, kind: str) -> None:
    mods = state.setdefault("modules", {})
    mods[slug] = {
        "last_version": version,
        "last_kind": kind,
        "last_checked": int(time.time()),
    }


def git_sync(url: str, dest: str) -> None:
    safe_mkdir(os.path.dirname(dest))
    if os.path.isdir(dest) and os.path.isdir(os.path.join(dest, ".git")):
        subprocess.check_call(["git", "-C", dest, "pull", "--ff-only"])
    else:
        subprocess.check_call(["git", "clone", "--depth", "1", url, dest])


def forge_list_modules(api_base: str, limit: int, offset: int) -> Dict[str, Any]:
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
    if "-" in slug:
        owner, name = slug.split("-", 1)
    else:
        owner, name = slug, "unknown"
    base_dir = os.path.join(dest_root, owner, name)
    return owner, name, base_dir


def discover_and_sync_forge(
    api_base: str,
    dest_root: str,
    include_globs: List[str],
    state_file: str,
    limit_per_page: int = 50,
    max_modules_seen: int = 500,
    max_modules_synced: int = 50,
    request_delay_sec: float = 0.5,
    download_delay_sec: float = 1.0,
    only_with_repo: bool = True,
    only_owner: Optional[str] = None,
    allowlist: Optional[List[str]] = None,
    denylist: Optional[List[str]] = None,
    index_unchanged: bool = False,
    check_interval_hours: int = 24,
) -> List[str]:
    safe_mkdir(dest_root)

    state = load_state(state_file)

    allowset = set(allowlist or [])
    denyset = set(denylist or [])

    synced_dirs: List[str] = []
    seen = 0
    synced = 0
    offset = 0

    while seen < max_modules_seen and synced < max_modules_synced:
        page = forge_list_modules(api_base, limit_per_page, offset)
        results = page.get("results") or []
        if not results:
            break

        for item in results:
            if seen >= max_modules_seen or synced >= max_modules_synced:
                break

            slug = item.get("slug")
            if not slug:
                continue
            seen += 1

            if allowset and slug not in allowset:
                continue
            if slug in denyset:
                continue
            if only_owner and not slug.startswith(f"{only_owner}-"):
                continue

            prev = get_state_entry(state, slug)
            if prev and check_interval_hours > 0:
                _, _, last_checked = prev
                if last_checked:
                    age = int(time.time()) - int(last_checked)
                    if age < check_interval_hours * 3600:
                        continue

            try:
                mod = forge_get_module(api_base, slug)
            except Exception as e:
                print(f"[forge] ERROR fetching {slug}: {e}")
                time.sleep(request_delay_sec)
                continue

            current = mod.get("current_release") or {}
            version = (current.get("version") or "").strip() or "unknown"
            metadata = current.get("metadata") or {}
            repo_url = (metadata.get("source") or "").strip()
            file_uri = (current.get("file_uri") or "").strip()

            use_git = is_probable_git_repo(repo_url)

            if only_with_repo and not use_git:
                upsert_state_entry(state, slug, version, "tar" if file_uri else "unknown")
                save_state(state_file, state)
                time.sleep(request_delay_sec)
                continue

            prev = get_state_entry(state, slug)
            if prev and prev[0] == version:
                upsert_state_entry(state, slug, version, prev[1] or ("git" if use_git else "tar"))
                save_state(state_file, state)

                if index_unchanged:
                    _, _, base_dir = slug_to_paths(dest_root, slug)
                    if use_git:
                        repo_dir = os.path.join(base_dir, "repo")
                        if os.path.isdir(repo_dir):
                            synced_dirs.append(repo_dir)
                    else:
                        rel_dir = os.path.join(base_dir, "releases", version)
                        if os.path.isdir(rel_dir):
                            synced_dirs.append(rel_dir)

                time.sleep(request_delay_sec)
                continue

            _, _, base_dir = slug_to_paths(dest_root, slug)

            try:
                if use_git:
                    repo_dir = os.path.join(base_dir, "repo")
                    git_sync(repo_url, repo_dir)
                    upsert_state_entry(state, slug, version, "git")
                    save_state(state_file, state)
                    synced_dirs.append(repo_dir)
                else:
                    if not file_uri:
                        upsert_state_entry(state, slug, version, "unknown")
                        save_state(state_file, state)
                        time.sleep(request_delay_sec)
                        continue

                    dl_dir = os.path.join(base_dir, "downloads")
                    safe_mkdir(dl_dir)
                    tgz_path = os.path.join(dl_dir, f"{slug}-{version}.tar.gz")
                    forge_download_file(api_base, file_uri, tgz_path)

                    rel_dir = os.path.join(base_dir, "releases", version)
                    safe_extract_tgz(tgz_path, rel_dir)

                    upsert_state_entry(state, slug, version, "tar")
                    save_state(state_file, state)
                    synced_dirs.append(rel_dir)

                synced += 1
                time.sleep(download_delay_sec)

            except Exception as e:
                print(f"[forge] ERROR syncing {slug}: {e}")
                time.sleep(request_delay_sec)

        offset += limit_per_page
        time.sleep(request_delay_sec)

    # dedupe
    uniq: List[str] = []
    s = set()
    for d in synced_dirs:
        if d not in s:
            uniq.append(d)
            s.add(d)

    print(f"[forge] run summary: seen={seen}, synced={synced}, returned_dirs={len(uniq)}")
    return uniq
