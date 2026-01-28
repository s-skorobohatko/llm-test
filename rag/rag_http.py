from fastapi import FastAPI
from pydantic import BaseModel
import subprocess

app = FastAPI()


class AskReq(BaseModel):
    question: str
    topk: int = 6
    minsim: float = 0.18


@app.post("/ask")
def ask(req: AskReq):
    cmd = [
        "python",
        "ask.py",
        "--topk",
        str(req.topk),
        "--minsim",
        str(req.minsim),
        req.question,
    ]
    out = subprocess.check_output(cmd, text=True)
    return {"answer": out}
