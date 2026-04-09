# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SRE Incident Response Environment Client."""

from typing import Any, Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ActionType, SreIncidentAction, SreIncidentObservation


class SreIncidentEnv(EnvClient[SreIncidentAction, SreIncidentObservation, State]):
    """
    Client for the SRE Incident Response Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with SreIncidentEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset(task_id="easy")
        ...     print(result.observation.title)
        ...
        ...     result = client.step(SreIncidentAction(
        ...         action_type=ActionType.LIST_ALERTS,
        ...         target="",
        ...         reasoning="Check active alerts"
        ...     ))
        ...     print(result.observation.action_result)

    Example with task selection:
        >>> with SreIncidentEnv(base_url="http://localhost:7860") as client:
        ...     # Run easy task
        ...     result = client.reset(task_id="easy")
        ...     ...
        ...     # Or run medium task
        ...     result = client.reset(task_id="medium")
        ...     ...
        ...     # Or run hard task
        ...     result = client.reset(task_id="hard")
    """

    def _step_payload(self, action: SreIncidentAction) -> Dict[str, Any]:
        """
        Convert SreIncidentAction to JSON payload for step message.

        Args:
            action: SreIncidentAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type.value
            if hasattr(action.action_type, "value")
            else str(action.action_type),
            "target": action.target,
            "reasoning": action.reasoning,
        }

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[SreIncidentObservation]:
        """
        Parse server response into StepResult[SreIncidentObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SreIncidentObservation
        """
        obs_data = payload.get("observation", {})

        active_alerts = obs_data.get("active_alerts", [])
        if isinstance(active_alerts, str):
            active_alerts = [active_alerts]

        system_status = obs_data.get("system_status", {})
        if isinstance(system_status, list):
            system_status = {str(k): str(v) for k, v in system_status}

        timeline = obs_data.get("timeline", [])
        if isinstance(timeline, str):
            timeline = [timeline]

        observation = SreIncidentObservation(
            incident_id=obs_data.get("incident_id", ""),
            severity=obs_data.get("severity", "P2"),
            title=obs_data.get("title", ""),
            description=obs_data.get("description", ""),
            action_result=obs_data.get("action_result", ""),
            active_alerts=active_alerts,
            system_status=system_status,
            timeline=timeline,
            step=obs_data.get("step", 0),
            hint=obs_data.get("hint", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
