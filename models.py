# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the SRE Incident Response Environment.

Defines all Pydantic models for actions, observations, task configurations,
and grading results used by the SRE incident response simulation.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation


class ActionType(str, Enum):
    """Available actions for SRE incident response."""

    LIST_ALERTS = "list_alerts"
    CHECK_DASHBOARD = "check_dashboard"
    RUN_QUERY = "run_query"
    GET_DEPLOYMENT = "get_deployment"
    ROLLBACK = "rollback"
    SCALE_SERVICE = "scale_service"
    RESTART_SERVICE = "restart_service"
    TOGGLE_FEATURE = "toggle_feature"
    PAGE_TEAM = "page_team"
    POST_UPDATE = "post_update"
    RESOLVE = "resolve"
    ESCALATE = "escalate"
    WAIT = "wait"


class SreIncidentAction(Action):
    """Action for the SRE Incident Response environment."""

    action_type: ActionType = Field(..., description="Type of action to perform")
    target: str = Field(
        default="", description="Target of the action (service name, query, etc.)"
    )
    reasoning: str = Field(
        default="", description="Agent's reasoning for taking this action"
    )


class SreIncidentObservation(Observation):
    """Observation from the SRE Incident Response environment."""

    incident_id: str = Field(default="", description="Unique incident identifier")
    severity: str = Field(default="P2", description="Incident severity (P1-P5)")
    title: str = Field(default="", description="Incident title")
    description: str = Field(default="", description="Incident description")
    action_result: str = Field(
        default="", description="Result of the last action taken"
    )
    active_alerts: List[str] = Field(
        default_factory=list, description="Currently active alerts"
    )
    system_status: Dict[str, str] = Field(
        default_factory=dict, description="System component status"
    )
    timeline: List[str] = Field(default_factory=list, description="Action timeline")
    step: int = Field(default=0, description="Current step number")
    hint: str = Field(default="", description="Optional hint for agent")
    done: bool = Field(default=False, description="Whether incident is resolved")
    reward: float = Field(default=0.0, description="Reward for this step")


class TaskConfig(BaseModel):
    """Configuration for a single task."""

    task_id: str = Field(..., description="Unique task identifier")
    task_name: str = Field(..., description="Human-readable task name")
    difficulty: str = Field(..., description="Difficulty level (easy/medium/hard)")
    description: str = Field(..., description="Task description")
    metric: str = Field(default="score", description="Metric used for evaluation")
    max_steps: int = Field(..., description="Maximum allowed steps")
    success_threshold: float = Field(
        default=0.6, description="Score threshold for success"
    )


class GraderResult(BaseModel):
    """Result from the grader endpoint."""

    task_id: str = Field(..., description="Task that was graded")
    score: float = Field(..., ge=0.0, le=1.0, description="Score in [0.0, 1.0] range")
    passed: bool = Field(..., description="Whether task passed the threshold")
    details: Dict[str, Any] = Field(default_factory=dict, description="Score breakdown")
    graded_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp of grading",
    )


class GradeRequest(BaseModel):
    """Request model for the grader endpoint."""

    task_id: str = Field(..., description="Task ID to grade")
    episode_id: Optional[str] = Field(default=None, description="Episode ID (optional)")
