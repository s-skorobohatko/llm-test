def _contains_unified_hunks(text: str) -> bool:
    for line in text.splitlines():
        if line.startswith("@@ "):
            return True
    return False


def _repair_diff_format(
    ollama: OllamaClient,
    *,
    model: str,
    num_ctx: int,
    num_predict: int,
    timeout_sec: int,
    system: str,
    diff_prompt: str,
    bad_output: str,
) -> str:
    """
    One-shot repair if model returned @@ hunks / wrong format.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": diff_prompt},
        {"role": "assistant", "content": bad_output},
        {
            "role": "user",
            "content": (
                "FORMAT ERROR: You output unified diff hunks (lines starting with @@). "
                "Re-output using ONLY the required format with NEW: full file content per DIFF FILE block. "
                "Do not include @@ hunks."
            ),
        },
    ]
    return ollama.chat(
        model=model,
        messages=messages,
        num_ctx=num_ctx,
        num_predict=num_predict,
        temperature=0.1,
        timeout_sec=timeout_sec,
    ).strip()
