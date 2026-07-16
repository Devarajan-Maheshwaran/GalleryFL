import asyncio
import httpx
import pytest

SERVER_URL = "http://localhost:8080"

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
            json={"device_model": "Test", "nickname": "TestUser"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "client_id" in data
        assert data["status"] == "registered"
        assert "model_version" in data

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
