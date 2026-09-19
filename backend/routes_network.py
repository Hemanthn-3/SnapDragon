"""
NEXUS Phase 11: Network Status & Local-Only Telemetry API Routes
Provides endpoints for inspecting strict air-gap compliance, network status,
and empirical network request counters.
"""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.logger import get_logger
from backend.network.guard import network_guard

logger = get_logger("nexus.routes_network")

network_router = APIRouter(prefix="/network", tags=["Strict Local-Only & Network Status"])


class SimulateOfflineRequest(BaseModel):
    offline: bool = Field(default=True, description="Enable or disable simulated network disconnect")


@network_router.get(
    "/status",
    summary="Get Strict Local-Only & Network Status",
    description="Returns real-time status of local air-gap enforcement, network interfaces, and empirical request counters.",
)
def get_network_status() -> Dict[str, Any]:
    return network_guard.detect_network_status()


@network_router.post(
    "/simulate-offline",
    summary="Toggle Simulated Network Disconnect",
    description="Sets simulated offline state for offline testing and verification.",
)
def toggle_simulate_offline(request: SimulateOfflineRequest) -> Dict[str, Any]:
    network_guard.set_simulated_offline(request.offline)
    return {
        "status": "SUCCESS",
        "simulated_offline": request.offline,
        "network_status": network_guard.detect_network_status(),
    }
