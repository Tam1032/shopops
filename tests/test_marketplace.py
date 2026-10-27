async def test_reply_then_conflict(market, store):
    r = await market.post("/reviews/R001/reply", json={"text": "Cảm ơn bạn!"})
    assert r.status_code == 200 and store.reviews["R001"]["status"] == "replied"
    again = await market.post("/reviews/R001/escalate", json={"reason": "x", "priority": "normal"})
    assert again.status_code == 409


async def test_unknown_order_is_404(market):
    assert (await market.get("/orders/NOPE")).status_code == 404


async def test_release_moves_backlog():
    import httpx
    from shopops.marketplace import Store, create_app

    app = create_app(Store(released=2), webhook_url="")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://m") as http:
        assert (await http.get("/reviews")).json()["total"] == 2
        assert (await http.get("/reviews/R003")).status_code == 404
        out = (await http.post("/admin/release", params={"n": 1})).json()
        assert out["released"] == ["R003"]
        assert (await http.get("/reviews")).json()["total"] == 3
        reset = (await http.post("/admin/reset", params={"released": 2})).json()
        assert reset["backlog"] == 30
        assert (await http.post("/admin/release", params={"n": 1})).json()["released"] == ["R003"]
