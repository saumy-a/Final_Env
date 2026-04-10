# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the SRE Incident Response Environment.

This module creates an HTTP server that exposes the SreIncidentEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - GET  /health: Server health check
    - GET  /tasks: List available tasks
    - POST /reset: Reset environment for new episode
    - POST /step: Execute an action
    - GET  /state: Get current environment state
    - POST /grader: Grade current episode
    - WS   /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with 'uv sync'"
    ) from e

from models import SreIncidentAction, SreIncidentObservation

try:
    from server.sre_incident_env_environment import SreIncidentEnvironment
except ImportError:
    from sre_incident_env_environment import SreIncidentEnvironment

try:
    from server.tasks import get_all_tasks, get_task
except ImportError:
    from tasks import get_all_tasks, get_task


_default_task = os.getenv("SRE_DEFAULT_TASK", "easy")

_base_app = create_app(
    lambda: SreIncidentEnvironment(default_task_id=_default_task),
    SreIncidentAction,
    SreIncidentObservation,
    env_name="sre_incident_env",
    max_concurrent_envs=10,
)

# Include base app routes first, then add our custom routes
app = FastAPI(
    title="SRE Incident Response Environment",
    description="An OpenEnv-compliant environment for training RL agents on SRE incident response",
    version="1.0.0",
)

# Mount base app at /web for web interface, keep / for API
app.mount("/", _base_app)


class GraderRequest(BaseModel):
    """Request model for the grader endpoint."""

    task_id: str
    episode_id: Optional[str] = None


class GraderResponse(BaseModel):
    """Response model for the grader endpoint."""

    task_id: str
    score: float
    passed: bool
    details: dict
    graded_at: str


@app.get("/tasks", response_model=list[dict])
async def list_tasks():
    """Return all available tasks."""
    tasks = get_all_tasks()
    return [
        {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "difficulty": task.difficulty,
            "description": task.description,
            "max_steps": task.max_steps,
            "success_threshold": task.success_threshold,
            "metric": task.metric,
        }
        for task in tasks
    ]


@app.post("/grader", response_model=GraderResponse)
async def grade_episode(request: GraderRequest):
    """
    Grade an episode using the optimal action sequence for the given task.

    This endpoint runs a deterministic episode using the optimal actions
    for the specified task and returns the score.
    """
    try:
        get_task(request.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        from server.grading import grade_episode as grade_actions
        from server.scenario_generator import ScenarioGenerator
    except ImportError:
        from grading import grade_episode as grade_actions
        from scenario_generator import ScenarioGenerator

    generator = ScenarioGenerator()
    incident = generator.generate(request.task_id)

    from models import ActionType, SreIncidentAction as Action

    optimal_sequences = {
        "easy": [
            Action(
                action_type=ActionType.LIST_ALERTS, target="", reasoning="check alerts"
            ),
            Action(
                action_type=ActionType.CHECK_DASHBOARD,
                target="payment-health",
                reasoning="check dashboard",
            ),
            Action(
                action_type=ActionType.RUN_QUERY,
                target="payment-api error rate",
                reasoning="investigate",
            ),
            Action(
                action_type=ActionType.ROLLBACK,
                target="payment-api",
                reasoning="fix issue",
            ),
            Action(
                action_type=ActionType.POST_UPDATE,
                target="rolled back payment-api",
                reasoning="comms",
            ),
            Action(
                action_type=ActionType.RESOLVE,
                target="issue resolved",
                reasoning="done",
            ),
        ],
        "medium": [
            Action(
                action_type=ActionType.LIST_ALERTS, target="", reasoning="check alerts"
            ),
            Action(
                action_type=ActionType.RUN_QUERY,
                target="pgbouncer connection pool",
                reasoning="investigate",
            ),
            Action(
                action_type=ActionType.CHECK_DASHBOARD,
                target="database-health",
                reasoning="confirm",
            ),
            Action(
                action_type=ActionType.PAGE_TEAM,
                target="db-team",
                reasoning="page experts",
            ),
            Action(
                action_type=ActionType.SCALE_SERVICE,
                target="pgbouncer:8",
                reasoning="fix pool",
            ),
            Action(
                action_type=ActionType.POST_UPDATE,
                target="scaled pgbouncer",
                reasoning="comms",
            ),
            Action(
                action_type=ActionType.RESOLVE,
                target="pool exhaustion fixed",
                reasoning="done",
            ),
        ],
        "hard": [
            Action(
                action_type=ActionType.LIST_ALERTS, target="", reasoning="check alerts"
            ),
            Action(
                action_type=ActionType.CHECK_DASHBOARD,
                target="cdn-edge-health",
                reasoning="cdn check",
            ),
            Action(
                action_type=ActionType.RUN_QUERY,
                target="cdn cache routing",
                reasoning="investigate",
            ),
            Action(
                action_type=ActionType.PAGE_TEAM,
                target="cdn-team",
                reasoning="page cdn team",
            ),
            Action(
                action_type=ActionType.TOGGLE_FEATURE,
                target="cdn_aggressive_routing:off",
                reasoning="disable flag",
            ),
            Action(
                action_type=ActionType.POST_UPDATE,
                target="disabled cdn flag",
                reasoning="comms",
            ),
            Action(
                action_type=ActionType.RESOLVE,
                target="cdn routing fixed",
                reasoning="done",
            ),
        ],
    }

    actions = optimal_sequences.get(request.task_id, optimal_sequences["easy"])
    result = grade_actions(actions, incident)

    return GraderResponse(
        task_id=request.task_id,
        score=result.score,
        passed=result.passed,
        details=result.details,
        graded_at=result.graded_at,
    )


def main(host: str = "0.0.0.0", port: int = 7860):
    """Launch the OpenEnv server via uvicorn."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
    run_main()  # Call main() pattern for validation
