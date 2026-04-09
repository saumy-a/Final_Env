# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SRE Incident Response Environment Implementation.

A realistic SRE incident response simulation where an RL agent must:
1. Triage and investigate production incidents
2. Identify root causes
3. Apply correct remediations
4. Coordinate with appropriate teams
5. Communicate status updates
6. Resolve incidents

The environment supports 3 difficulty levels with 5 varied incidents each,
providing diverse training scenarios for LLM agents.
"""

import uuid
from typing import Dict, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        GraderResult,
        SreIncidentAction,
        SreIncidentObservation,
        TaskConfig,
    )
except ImportError:
    from models import (
        GraderResult,
        SreIncidentAction,
        SreIncidentObservation,
        TaskConfig,
    )

try:
    from .grading import EpisodeGrader
    from .scenario_generator import Incident, ScenarioGenerator
    from .tasks import get_all_tasks, get_task
except ImportError:
    from grading import EpisodeGrader
    from scenario_generator import Incident, ScenarioGenerator
    from tasks import get_all_tasks, get_task


class SreIncidentEnvironment(Environment):
    """
    SRE Incident Response Environment.

    Simulates real-world incident response scenarios where an agent must:
    - List and analyze active alerts
    - Check dashboards for metrics
    - Query logs and metrics
    - Apply remediations (rollback, scale, restart, toggle)
    - Page appropriate teams
    - Post status updates
    - Resolve incidents

    Attributes:
        SUPPORTS_CONCURRENT_SESSIONS: Whether multiple WebSocket sessions are supported
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, default_task_id: str = "easy", seed: Optional[int] = None):
        """Initialize the SRE Incident Environment.

        Args:
            default_task_id: Default task to use (easy/medium/hard)
            seed: Optional random seed for reproducibility
        """
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._default_task_id = default_task_id
        self._seed = seed
        self._scenario_generator = ScenarioGenerator(seed=seed)
        self._reset()

    def _reset(self) -> None:
        """Reset internal state for a new episode."""
        self._current_incident: Optional[Incident] = None
        self._current_task_id: Optional[str] = None
        self._actions_taken: List[SreIncidentAction] = []
        self._timeline: List[str] = []
        self._grader: Optional[EpisodeGrader] = None
        self._episode_complete = False
        self._episode_reward = 0.0
        self._cumulative_reward = 0.0

    def reset(
        self, task_id: Optional[str] = None, seed: Optional[int] = None
    ) -> SreIncidentObservation:
        """Reset the environment and start a new incident.

        Args:
            task_id: Task difficulty (easy/medium/hard). Defaults to self._default_task_id
            seed: Optional seed override for this episode

        Returns:
            SreIncidentObservation with initial incident details
        """
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._reset()

        task_id = task_id or self._default_task_id
        actual_seed = seed if seed is not None else self._seed

        self._current_task_id = task_id
        self._current_incident = self._scenario_generator.generate(
            task_id, seed=actual_seed
        )
        self._grader = EpisodeGrader(self._current_incident)

        initial_message = (
            f"New incident reported: {self._current_incident.title}\n"
            f"Severity: {self._current_incident.severity}\n"
            f"Description: {self._current_incident.description}\n\n"
            f"Affected Service: {self._current_incident.affected_service}\n"
            f"Available Actions: list_alerts, check_dashboard, run_query, "
            f"get_deployment, rollback, scale_service, restart_service, "
            f"toggle_feature, page_team, post_update, resolve, escalate, wait"
        )

        self._timeline.append(
            f"[Step 0] Incident started: {self._current_incident.incident_id}"
        )

        return SreIncidentObservation(
            incident_id=self._current_incident.incident_id,
            severity=self._current_incident.severity,
            title=self._current_incident.title,
            description=self._current_incident.description,
            action_result=initial_message,
            active_alerts=self._current_incident.symptoms,
            system_status={
                self._current_incident.affected_service: "DEGRADED",
                "monitoring": "FIRING",
            },
            timeline=self._timeline.copy(),
            step=0,
            hint=self._get_hint(0),
            done=False,
            reward=0.0,
        )

    def step(self, action: SreIncidentAction) -> SreIncidentObservation:
        """Execute one step in the incident response.

        Args:
            action: The action to take

        Returns:
            SreIncidentObservation with updated state and reward
        """
        self._state.step_count += 1
        self._actions_taken.append(action)

        result, reward = self._process_action(action)

        self._timeline.append(
            f"[Step {self._state.step_count}] {action.action_type.value}: {action.target}"
        )

        task = get_task(self._current_task_id)
        done = self._episode_complete or self._state.step_count >= task.max_steps

        if done and not self._episode_complete:
            self._episode_complete = True
            if isinstance(result, str) and "resolved" in result.lower():
                self._episode_reward = self._calculate_final_score()

        return SreIncidentObservation(
            incident_id=self._current_incident.incident_id,
            severity=self._current_incident.severity,
            title=self._current_incident.title,
            description=self._current_incident.description,
            action_result=result,
            active_alerts=self._current_incident.symptoms if not done else [],
            system_status=self._get_system_status(done),
            timeline=self._timeline.copy(),
            step=self._state.step_count,
            hint=self._get_hint(self._state.step_count) if not done else "",
            done=done,
            reward=reward,
        )

    def _process_action(self, action: SreIncidentAction) -> tuple[str, float]:
        """Process an action and return result and reward.

        Args:
            action: The action to process

        Returns:
            Tuple of (result_message, reward)
        """
        if self._grader is None:
            return "Error: Environment not initialized", 0.0

        action_type = (
            action.action_type.value
            if hasattr(action.action_type, "value")
            else str(action.action_type)
        )
        target = action.target

        evaluation = self._grader.evaluate_action(action)
        step_reward = evaluation.get("points", 0.0)

        if action_type == "list_alerts":
            alerts = "\n".join(f"  - {s}" for s in self._current_incident.symptoms)
            result = f"Active alerts:\n{alerts}"
            result += f"\n\nAffected service: {self._current_incident.affected_service}"
            result += f"\nSymptoms: {self._current_incident.symptoms[0]}"

        elif action_type == "check_dashboard":
            if target.lower() == self._current_incident.dashboard_name.lower():
                result = f"Dashboard '{target}' shows anomaly in {self._current_incident.affected_service}"
                result += "\nMetric spike detected matching root cause pattern"
                step_reward += 0.10
            else:
                result = f"Dashboard '{target}' shows normal metrics"

        elif action_type == "run_query":
            hint = self._current_incident.query_hint.lower()
            if hint in target.lower() or any(w in target.lower() for w in hint.split()):
                result = f"Query results for '{target}':"
                result += f"\n  - Found correlation with {self._current_incident.affected_service}"
                result += f"\n  - Root cause indicator: {self._current_incident.root_cause[:100]}"
                step_reward += 0.15
            else:
                result = f"Query results for '{target}': No significant anomalies found"

        elif action_type == "get_deployment":
            if target.lower() == self._current_incident.affected_service.lower():
                result = f"Recent deployment for {target}:"
                result += "\n  - Deployment 2 hours ago"
                result += "\n  - Changes: configuration update"
                step_reward += 0.05
            else:
                result = f"No recent deployments for {target}"

        elif action_type == "rollback":
            if target.lower() == self._current_incident.affected_service.lower():
                result = f"Rollback initiated for {target}..."
                result += "\nRollback completed successfully"
                result += "\nService health: RECOVERING"
                self._episode_complete = True
                step_reward += 0.35
            else:
                result = f"Rollback for {target} completed (may not be the root cause)"

        elif action_type == "scale_service":
            if target.lower().startswith(
                self._current_incident.affected_service.lower()
            ):
                replicas = target.split(":")[-1] if ":" in target else "5"
                result = f"Scaling {self._current_incident.affected_service} to {replicas} replicas..."
                result += "\nScale operation completed"
                result += "\nService health: RECOVERING"
                self._episode_complete = True
                step_reward += 0.35
            else:
                result = f"Scaled {target} (may not resolve the issue)"

        elif action_type == "restart_service":
            if target.lower() == self._current_incident.affected_service.lower():
                result = f"Restarting {target}..."
                result += "\nService restarted successfully"
                result += "\nService health: RECOVERING"
                self._episode_complete = True
                step_reward += 0.35
            else:
                result = f"Restarted {target} (may not resolve the issue)"

        elif action_type == "toggle_feature":
            if "off" in target.lower():
                result = f"Toggling feature flag: {target}"
                result += "\nFeature disabled"
                result += "\nService health: RECOVERING"
                self._episode_complete = True
                step_reward += 0.35
            else:
                result = f"Toggled feature: {target}"

        elif action_type == "page_team":
            result = f"Page sent to {target}"
            if (
                self._current_incident.team_to_page
                and target.lower() == self._current_incident.team_to_page.lower()
            ):
                result += "\nTeam acknowledged and is investigating"
                step_reward += 0.15
            else:
                result += "\nTeam will respond shortly"

        elif action_type == "post_update":
            result = "Status update posted to incident channel:"
            result += f"\n  - Incident: {self._current_incident.title}"
            result += "\n  - Status: Investigating"
            result += f"\n  - Service: {self._current_incident.affected_service}"

        elif action_type == "resolve":
            if self._grader._remediation_done:
                result = f"Incident {self._current_incident.incident_id} RESOLVED"
                result += "\nRoot cause: " + self._current_incident.root_cause
                result += "\nAll systems nominal"
                self._episode_complete = True
                step_reward += 0.15
            else:
                result = (
                    "Cannot resolve without proper remediation. Continue investigating."
                )

        elif action_type == "escalate":
            result = "Incident escalated to management"
            self._episode_complete = True

        elif action_type == "wait":
            result = "Waiting for more data..."
            step_reward = -0.01

        else:
            result = f"Action {action_type} executed on {target}"
            step_reward = 0.0

        return result, step_reward

    def _get_hint(self, step: int) -> str:
        """Get a hint based on current step.

        Args:
            step: Current step number

        Returns:
            Hint string or empty string
        """
        if step < 3:
            return ""
        elif step < 6:
            return f"Hint: Start by listing alerts or checking the {self._current_incident.dashboard_name} dashboard"
        elif step < 10:
            return "Hint: Try running a query to investigate the root cause"
        elif step < 15:
            return (
                "Hint: Consider applying a remediation or paging the appropriate team"
            )
        return "Hint: Make sure to post an update and resolve the incident"

    def _get_system_status(self, done: bool) -> Dict[str, str]:
        """Get current system status.

        Args:
            done: Whether episode is complete

        Returns:
            Dictionary of system component statuses
        """
        if done:
            return {
                self._current_incident.affected_service: "HEALTHY",
                "monitoring": "OK",
            }
        return {
            self._current_incident.affected_service: "DEGRADED",
            "monitoring": "FIRING",
        }

    def _calculate_final_score(self) -> float:
        """Calculate the final episode score.

        Returns:
            Final score in [0, 1] range
        """
        if self._grader is None:
            return 0.0
        result = self._grader.compute_score()
        return result.score

    def grade(self, task_id: Optional[str] = None) -> GraderResult:
        """Grade the current episode.

        Args:
            task_id: Optional task ID override

        Returns:
            GraderResult with score and breakdown
        """
        if self._grader is None:
            return GraderResult(
                task_id=task_id or self._current_task_id or "unknown",
                score=0.0,
                passed=False,
                details={"error": "No episode recorded"},
            )
        return self._grader.compute_score()

    def get_tasks(self) -> List[TaskConfig]:
        """Get all available tasks.

        Returns:
            List of TaskConfig objects
        """
        return get_all_tasks()

    def get_incident(self) -> Optional[Incident]:
        """Get the current incident.

        Returns:
            Current Incident or None
        """
        return self._current_incident

    @property
    def state(self) -> State:
        """Get current environment state.

        Returns:
            Current State object
        """
        return self._state
