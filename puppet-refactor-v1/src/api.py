from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.refactor_engine import refactor_module

app = FastAPI(title="Puppet Refactor v1", version="1.0")


class RefactorReq(BaseModel):
    module_path: str
    task: str
    cfg_path: str = "config.yaml"


@app.post("/refactor")
def refactor(req: RefactorReq):
    try:
        res = refactor_module(req.module_path, req.task, cfg_path=req.cfg_path)
        return {"ok": True, "plan": res.plan, "diff": res.diff, "report_md": res.report_md}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
