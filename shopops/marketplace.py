"""Mock marketplace API standing in for Shopee / TikTok Shop / Amazon seller APIs.

Real seller APIs need an approved seller account, so the agent talks to this
service instead. The routes mirror what the real APIs offer (reviews, orders,
stock, reply to review); swapping in a real API means rewriting
`shopops/mcp_server.py`'s HTTP calls, nothing else.
"""

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class Store:
    def __init__(self, data_dir: Path = DATA_DIR, released: int | None = None):
        catalog = json.loads((data_dir / "catalog.json").read_text(encoding="utf-8"))
        self.products = {p["sku"]: p for p in catalog["products"]}
        self.orders = {o["order_id"]: o for o in catalog["orders"]}
        reviews = [json.loads(line) for line in (data_dir / "reviews.jsonl").read_text(encoding="utf-8").splitlines() if line]
        for r in reviews:
            r.update(status="new", reply=None, escalation=None)
        self.reviews = {r["review_id"]: r for r in reviews}
        # Reviews not yet "posted" by customers; /admin/release moves them in one at a time
        # so the n8n schedule or webhook has something new to pick up.
        ids = list(self.reviews)
        cut = len(ids) if released is None else released
        self.backlog = ids[cut:]
        for rid in self.backlog:
            self.reviews[rid]["status"] = "unreleased"


class ReplyIn(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


class EscalationIn(BaseModel):
    reason: str = Field(min_length=1)
    priority: str = Field(pattern="^(normal|high)$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def notify(url: str, reviews: list[dict]) -> None:
    """Webhook calls go one at a time, after the response, so each subscriber run finishes
    before the next starts (parallel n8n runs appending to the same sheet overwrite each other)."""
    async with httpx.AsyncClient(timeout=120) as client:
        for r in reviews:
            try:
                await client.post(url, json={"event": "review.created", "review": r})
            except httpx.HTTPError:
                pass  # a down subscriber must not break the marketplace


def create_app(store: Store | None = None, webhook_url: str | None = None) -> FastAPI:
    app = FastAPI(title="Mock marketplace", version="1.0")
    app.state.store = store or Store(released=int(os.getenv("RELEASED_REVIEWS", "8")))
    app.state.webhook_url = webhook_url if webhook_url is not None else os.getenv("REVIEW_WEBHOOK_URL")

    def st() -> Store:
        return app.state.store

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/reviews")
    def list_reviews(status: str = "new", limit: int = 20):
        items = [r for r in st().reviews.values() if r["status"] == status]
        return {"items": items[:limit], "total": len(items)}

    @app.get("/reviews/{review_id}")
    def get_review(review_id: str):
        r = st().reviews.get(review_id)
        if r is None or r["status"] == "unreleased":
            raise HTTPException(404, f"review {review_id} not found")
        return r

    @app.get("/orders/{order_id}")
    def get_order(order_id: str):
        o = st().orders.get(order_id)
        if o is None:
            raise HTTPException(404, f"order {order_id} not found")
        return o

    @app.get("/inventory/{sku}")
    def get_inventory(sku: str):
        p = st().products.get(sku)
        if p is None:
            raise HTTPException(404, f"sku {sku} not found")
        return {"sku": sku, "name": p["name"], "stock": p["variants"]}

    def _open_review(review_id: str) -> dict:
        r = get_review(review_id)
        if r["status"] != "new":
            raise HTTPException(409, f"review {review_id} already {r['status']}")
        return r

    @app.post("/reviews/{review_id}/reply")
    def reply(review_id: str, body: ReplyIn):
        r = _open_review(review_id)
        r.update(status="replied", reply={"text": body.text, "at": _now()})
        return r

    @app.post("/reviews/{review_id}/escalate")
    def escalate(review_id: str, body: EscalationIn):
        r = _open_review(review_id)
        r.update(status="escalated", escalation={**body.model_dump(), "at": _now()})
        return r

    @app.post("/admin/release")
    async def release(background: BackgroundTasks, n: int = 1):
        """Simulate customers posting n new reviews; fires the webhook for each one."""
        released = []
        while st().backlog and len(released) < n:
            r = st().reviews[st().backlog.pop(0)]
            r["status"] = "new"
            released.append(r)
        if app.state.webhook_url:
            background.add_task(notify, app.state.webhook_url, released)
        return {"released": [r["review_id"] for r in released], "remaining": len(st().backlog)}

    @app.post("/admin/reset")
    def reset(released: int | None = None):
        """Back to the starting state: the first `released` reviews posted, the rest in the backlog."""
        if released is None:
            released = int(os.getenv("RELEASED_REVIEWS", "8"))
        app.state.store = Store(released=released)
        return {"ok": True, "released": released, "backlog": len(app.state.store.backlog)}

    return app


def fresh_store_with(review_ids: list[str] | None = None) -> Store:
    """A full copy of the seed data, used by evals so every case starts clean."""
    s = Store(released=None)
    if review_ids is not None:
        s.reviews = {k: copy.deepcopy(v) for k, v in s.reviews.items() if k in review_ids}
    return s


app = create_app()
