"""Agent API that n8n (or anything else) calls.

The agent never posts or escalates on its own here: side-effect tool calls are
queued as pending actions, and a person approves them (from a Google Sheet via
n8n, or with a plain HTTP call). Approval may edit the reply text first.

    POST /triage/{review_id}        run the agent, return its labels + pending actions
    GET  /actions?status=pending    list actions waiting for approval
    POST /actions/{id}/approve      run the action (optionally with edited arguments)
    POST /actions/{id}/reject
    POST /webhook/review            marketplace webhook: triage new reviews in the background
"""

import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from mcp import Client
from pydantic import BaseModel

from shopops.agent import build_agent, triage
from shopops.harness.loop import DeferToQueue
from shopops.harness.mcp_tools import load_mcp_tools
from shopops.harness.trace import Tracer
from shopops.mcp_server import build_server

log = logging.getLogger("shopops.api")


class ActionStore:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, review_id TEXT, tool TEXT, arguments TEXT,
            labels TEXT, status TEXT, result TEXT, created_at TEXT, decided_at TEXT)""")
        self.db.commit()

    def add(self, review_id: str, tool: str, arguments: dict, labels: dict | None) -> int:
        cur = self.db.execute(
            "INSERT INTO actions (review_id, tool, arguments, labels, status, created_at) VALUES (?,?,?,?,?,?)",
            (review_id, tool, json.dumps(arguments, ensure_ascii=False), json.dumps(labels, ensure_ascii=False),
             "pending", _now()))
        self.db.commit()
        return cur.lastrowid

    def get(self, action_id: int) -> dict | None:
        row = self.db.execute("SELECT * FROM actions WHERE id = ?", (action_id,)).fetchone()
        return _row(row) if row else None

    def list(self, status: str | None) -> list[dict]:
        q, args = "SELECT * FROM actions", ()
        if status:
            q, args = q + " WHERE status = ?", (status,)
        return [_row(r) for r in self.db.execute(q + " ORDER BY id", args)]

    def decide(self, action_id: int, status: str, result: str | None, arguments: dict | None = None) -> None:
        if arguments is not None:
            self.db.execute("UPDATE actions SET arguments = ? WHERE id = ?",
                            (json.dumps(arguments, ensure_ascii=False), action_id))
        self.db.execute("UPDATE actions SET status = ?, result = ?, decided_at = ? WHERE id = ?",
                        (status, result, _now(), action_id))
        self.db.commit()


def _row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["arguments"] = json.loads(d["arguments"])
    d["labels"] = json.loads(d["labels"]) if d["labels"] else None
    return d


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Approval(BaseModel):
    arguments: dict | None = None  # e.g. {"text": "edited reply"}; merged over the proposed arguments


def make_llm():
    if os.getenv("LLM_MODEL"):
        from shopops.harness.llm import OpenAICompatLLM
        return OpenAICompatLLM.from_env("LLM")
    from shopops.baseline import KeywordBaseline
    log.warning("LLM_MODEL is not set; using the keyword baseline")
    return KeywordBaseline()


def create_app(marketplace: httpx.AsyncClient | None = None, db_path: str | None = None, llm=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = marketplace or httpx.AsyncClient(
            base_url=os.getenv("MARKETPLACE_URL", "http://localhost:8000"), timeout=15)
        app.state.actions = ActionStore(db_path or os.getenv("ACTIONS_DB", "actions.sqlite"))
        app.state.llm = llm or make_llm()
        yield
        if marketplace is None:
            await app.state.http.aclose()

    app = FastAPI(title="ShopOps agent", version="1.0", lifespan=lifespan)

    def mcp_client() -> Client:
        # MCP_URL points at a running `python -m shopops.mcp_server --http`; otherwise run it in-process.
        url = os.getenv("MCP_URL")
        return Client(url) if url else Client(build_server(app.state.http))

    @app.get("/health")
    def health():
        return {"ok": True, "model": app.state.llm.name}

    @app.post("/triage/{review_id}")
    async def run_triage(review_id: str):
        queue = DeferToQueue()
        trace_dir = os.getenv("TRACE_DIR")
        tracer = Tracer(Path(trace_dir) / "traces.jsonl" if trace_dir else None, run_id=review_id)
        async with mcp_client() as mcp:
            agent = await build_agent(app.state.llm, mcp, confirmer=queue, tracer=tracer)
            run = await triage(agent, review_id)
        pending = [{"id": app.state.actions.add(review_id, a.tool, a.arguments, run.final),
                    "tool": a.tool, "arguments": a.arguments} for a in queue.queued]
        review = await app.state.http.get(f"/reviews/{review_id}")
        review_text = review.json().get("text") if review.status_code == 200 else None
        first = pending[0] if pending else None
        # Flat copy of the first pending action for low-code tools (n8n blocks `.arguments` in expressions).
        next_action = first and {"id": first["id"], "tool": first["tool"],
                                 "text": first["arguments"].get("text") or first["arguments"].get("reason")}
        return {"review_id": review_id, "review_text": review_text, "labels": run.final, "stop_reason": run.stop_reason,
                "steps": run.steps, "pending_actions": pending, "next_action": next_action,
                "tokens": run.usage.prompt_tokens + run.usage.completion_tokens}

    @app.get("/actions")
    def list_actions(status: str | None = "pending"):
        return {"items": app.state.actions.list(status)}

    @app.post("/actions/{action_id}/approve")
    async def approve(action_id: int, body: Approval | None = None):
        action = app.state.actions.get(action_id)
        if action is None:
            raise HTTPException(404, "no such action")
        if action["status"] != "pending":
            raise HTTPException(409, f"action already {action['status']}")
        args = {**action["arguments"], **((body and body.arguments) or {})}
        async with mcp_client() as mcp:
            tools = {t.name: t for t in await load_mcp_tools(mcp)}
            try:
                result = await tools[action["tool"]].handler(args)
            except Exception as exc:
                app.state.actions.decide(action_id, "failed", str(exc), args)
                raise HTTPException(502, f"{action['tool']} failed: {exc}")
        app.state.actions.decide(action_id, "done", result, args)
        return {"id": action_id, "status": "done", "result": json.loads(result)}

    @app.post("/actions/{action_id}/reject")
    def reject(action_id: int):
        if app.state.actions.get(action_id) is None:
            raise HTTPException(404, "no such action")
        app.state.actions.decide(action_id, "rejected", None)
        return {"id": action_id, "status": "rejected"}

    @app.post("/webhook/review")
    async def on_review(payload: dict, background: BackgroundTasks):
        review_id = payload.get("review", {}).get("review_id")
        if not review_id:
            raise HTTPException(422, "payload.review.review_id missing")
        background.add_task(run_triage, review_id)
        return {"accepted": review_id}

    return app


app = create_app()
