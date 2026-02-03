SYSTEM = (
    "You are a senior Puppet 8 engineer. "
    "Follow constraints exactly. Output must be strictly formatted."
)

COMMON = """MANDATORY GOOD PRACTICE RULES (Puppet 8):
- Role/Profile:
  - Roles only include profiles (no resources in roles).
  - Profiles use lookup() for data.
  - Modules remain generic; no site-specific values in components.
- Data types:
  - Every class parameter must have Puppet data types (String, Boolean, Integer, Optional[], Stdlib::Port, etc).
  - Do not use legacy validate_* functions.
- Idempotency:
  - exec must include onlyif/unless/creates as appropriate.
- Formatting:
  - ensure must be the first attribute in a resource.
  - align arrows (=>) within a block.

HARD CONSTRAINTS:
- You MUST ONLY reference and modify files listed in MODULE_FILES.
- You MUST NOT invent file paths.
- Output ONLY the required format. No extra text.
"""

PLAN_PROMPT = """{common}

TASK:
{task}

REFERENCE_CONTEXT (authoritative guidance):
{ref_ctx}

MODULE_CONTEXT (authoritative module files and content):
{module_ctx}

OUTPUT FORMAT:
1) FILES TO TOUCH (paths from MODULE_FILES)
2) PLAN (bullet points per file)
3) JUSTIFICATION (map plan bullets -> REF [n])
"""

DIFF_PROMPT = """{common}

TASK:
{task}

PLAN (must follow):
{plan}

REFERENCE_CONTEXT:
{ref_ctx}

MODULE_CONTEXT:
{module_ctx}

OUTPUT FORMAT (EXACT):
For each changed file, output:

DIFF FILE: <path>
--------------------------------
OLD:
<only the relevant original lines>

NEW:
<the modified lines>
--------------------------------
END DIFF

RULES:
- Only files from MODULE_FILES.
- One DIFF FILE block per file.
- Do NOT include unchanged files.
"""
