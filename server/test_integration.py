import asyncio
import httpx
import pytest

# Basic integration tests against a running server
# Assumes the server is running on localhost:8080

SERVER_URL = "http://localhost:8080"

@pytest.mark.asyncio
async def test_server_status():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{SERVER_URL}/api/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "is_training" in data
        assert "current_round" in data

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

@pytest.mark.asyncio
async def test_get_model():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{SERVER_URL}/api/model/current")
        assert resp.status_code == 200
        # The content should be a base64 string
        assert len(resp.content) > 100 
