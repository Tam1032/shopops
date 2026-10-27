import inspect

import httpx
import pytest
from mcp import Client

from shopops.marketplace import Store, create_app
from shopops.mcp_server import build_server


@pytest.fixture
def store():
    return Store(released=None)


@pytest.fixture
async def market(store):
    app = create_app(store, webhook_url="")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://marketplace") as http:
        yield http


@pytest.fixture
async def mcp(market):
    async with Client(build_server(market)) as client:
        yield client


@pytest.fixture
def anyio_backend():
    return "asyncio"


def pytest_collection_modifyitems(items):
    for item in items:
        if inspect.iscoroutinefunction(getattr(item, "function", None)):
            item.add_marker(pytest.mark.anyio)
