import httpx
import pytest
import json
import os

SERVER_URL = "http://localhost:8000"

# Read the live access token from config.json so the tests match whatever the
# server was actually started with (the default "fgt-pass" is overridden there).
_cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
try:
    with open(_cfg_path) as _f:
        TOKEN = json.load(_f).get("server_token", "fgt-pass")
except Exception:
    TOKEN = "fgt-pass"
HEADERS = {"X-FGT-Token": TOKEN}

@pytest.mark.asyncio
async def test_server_status():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{SERVER_URL}/api/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "is_training" in data
        assert "current_round" in data
        assert "total_parameters" in data

@pytest.mark.asyncio
async def test_register_client():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SERVER_URL}/api/register",
            json={"device_model": "Test", "nickname": "TestUser"},
            headers=HEADERS
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "client_id" in data
        assert data["status"] == "registered"
        assert "model_version" in data

@pytest.mark.asyncio
async def test_register_without_token():
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SERVER_URL}/api/register",
            json={"device_model": "Test", "nickname": "NoToken"}
        )
        assert resp.status_code == 401

@pytest.mark.asyncio
async def test_get_model():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{SERVER_URL}/api/model/current")
        assert resp.status_code == 200
        assert len(resp.content) > 100

@pytest.mark.asyncio
async def test_metrics_summary():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{SERVER_URL}/api/metrics/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_rounds" in data

@pytest.mark.asyncio
async def test_leaderboard():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{SERVER_URL}/api/metrics/leaderboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "leaderboard" in data


@pytest.mark.asyncio
async def test_tag_demand_signal_and_read():
    async with httpx.AsyncClient() as client:
        # No token -> rejected (consistency with other POST endpoints).
        r0 = await client.post(f"{SERVER_URL}/api/taxonomy/signal", json={"tags": ["beach", "food"]})
        assert r0.status_code == 401

        # With token: valid taxonomy leaves are recorded, unknown tags ignored.
        r1 = await client.post(
            f"{SERVER_URL}/api/taxonomy/signal",
            json={"client_id": "tester", "tags": ["beach", "food", "not_a_real_tag"]},
            headers=HEADERS,
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["ok"] is True
        assert d1["recorded"] == 2  # beach + food valid; not_a_real_tag ignored

        # Read demand back.
        r2 = await client.get(f"{SERVER_URL}/api/taxonomy/demand")
        assert r2.status_code == 200
        d2 = r2.json()
        assert "trending" in d2 and "weights" in d2
        assert d2["total_signals"] >= 1
        tags = [t["tag"] for t in d2["trending"]]
        assert "beach" in tags and "food" in tags
