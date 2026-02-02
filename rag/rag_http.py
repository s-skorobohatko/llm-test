from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import json

app = FastAPI()


class AskReq(BaseModel):
    question: str
    scope: str
    topk: int = 6
    minsim: float = 0.18
    two_pass: bool = True
    mode: str = "diff"  # diff | plan | both
    ref_source: list[str] = []
    prompt_dir: str | None = None


@app.post("/ask")
def ask(req: AskReq):
    cmd = [
        "python",
        "ask.py",
        "--json",
        "--scope", req.scope,
        "--topk", str(req.topk),
        "--minsim", str(req.minsim),
        "--mode", req.mode,
    ]

    if req.two_pass:
        cmd.append("--two_pass")

    for s in req.ref_source:
        cmd += ["--ref_source", s]

    if req.prompt_dir:
        cmd += ["--prompt_dir", req.prompt_dir]

    cmd.append(req.question)

    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return json.loads(out)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.output)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="ask.py did not return valid JSON")
