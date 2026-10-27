import httpx
import pytest

from shopops.api import create_app as create_api
from shopops.baseline import KeywordBaseline


@pytest.fixture
async def api(market, tmp_path):
    app = create_api(marketplace=market, db_path=str(tmp_path / "a.sqlite"), llm=KeywordBaseline())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as client:
            yield client


async def test_triage_queues_then_approve_with_edit(api, store):
    out = (await api.post("/triage/R003")).json()
    assert out["labels"]["topic"] == "delivery"
    [action] = out["pending_actions"]
    assert action["tool"] == "post_reply" and store.reviews["R003"]["status"] == "new"
    assert out["next_action"] == {"id": action["id"], "tool": "post_reply", "text": action["arguments"]["text"]}
    assert out["review_text"].startswith("Giao hàng")

    pending = (await api.get("/actions")).json()["items"]
    assert [a["id"] for a in pending] == [action["id"]]

    r = await api.post(f"/actions/{action['id']}/approve", json={"arguments": {"text": "Bản đã sửa"}})
    assert r.status_code == 200 and store.reviews["R003"]["reply"]["text"] == "Bản đã sửa"
    assert (await api.post(f"/actions/{action['id']}/approve")).status_code == 409


async def test_reject(api, store):
    out = (await api.post("/triage/R002")).json()
    [action] = out["pending_actions"]
    assert action["tool"] == "escalate_to_human"
    assert (await api.post(f"/actions/{action['id']}/reject")).json()["status"] == "rejected"
    assert store.reviews["R002"]["status"] == "new"
