# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Grading logic for SRE Incident Response Environment.

Provides deterministic scoring based on agent actions during incident resolution.
Score breakdown:
- Root cause investigation: 0.25
- Correct remediation: 0.35
- Team notification (if needed): 0.15
- Communication: 0.10
- Resolution: 0.15
- Efficiency bonus: up to 0.10
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from ..models import ActionType, GraderResult, SreIncidentAction
    from .scenario_generator import Incident
except ImportError:
    from models import ActionType, GraderResult, SreIncidentAction
    from scenario_generator import Incident


class EpisodeGrader:
    """Grades agent performance on SRE incident response tasks."""

    SCORE_INVESTIGATION = 0.25
    SCORE_REMEDIATION = 0.35
    SCORE_TEAM_NOTIFICATION = 0.15
    SCORE_COMMUNICATION = 0.10
    SCORE_RESOLUTION = 0.15
    MAX_EFFICIENCY_BONUS = 0.10
    EFFICIENCY_DECAY_RATE = 0.02

    WRONG_ACTION_PENALTY = 0.02
    MAX_WRONG_ACTION_PENALTY = 0.10

    def __init__(self, incident: Incident):
        """Initialize the grader with an incident.

        Args:
            incident: The incident scenario being graded
        """
        self.incident = incident
        self.actions_taken: List[SreIncidentAction] = []
        self.wrong_actions = 0
        self._investigation_done = False
        self._remediation_done = False
        self._team_notified = False
        self._team_should_be_notified = incident.team_to_page is not None
        self._communication_done = False
        self._resolution_done = False
        self._steps_taken = 0

    def record_action(self, action: SreIncidentAction) -> None:
        """Record an action taken by the agent.

        Args:
            action: The action taken
        """
        self.actions_taken.append(action)
        self._steps_taken += 1

    def evaluate_action(self, action: SreIncidentAction) -> Dict[str, Any]:
        """Evaluate a single action and return feedback.

        Args:
            action: The action to evaluate

        Returns:
            Dictionary with evaluation result and any reward
        """
        feedback = {"recognized": False, "points": 0.0, "message": ""}
        action_type = (
            action.action_type.value
            if hasattr(action.action_type, "value")
            else str(action.action_type)
        )

        if action_type == "list_alerts":
            feedback["recognized"] = True
            feedback["message"] = "Listed active alerts"
            self._investigation_done = True

        elif action_type == "check_dashboard":
            target = action.target.lower() if action.target else ""
            if target == self.incident.dashboard_name.lower():
                feedback["recognized"] = True
                feedback["points"] = 0.10
                feedback["message"] = (
                    f"Checked correct dashboard: {self.incident.dashboard_name}"
                )
                self._investigation_done = True
            elif target:
                feedback["recognized"] = True
                feedback["points"] = 0.02
                feedback["message"] = f"Checked dashboard: {target}"

        elif action_type == "run_query":
            target = action.target.lower() if action.target else ""
            hint = self.incident.query_hint.lower()
            if hint in target or any(word in target for word in hint.split()):
                feedback["recognized"] = True
                feedback["points"] = 0.15
                feedback["message"] = f"Queried relevant metrics: {action.target}"
                self._investigation_done = True
            elif target:
                feedback["recognized"] = True
                feedback["points"] = 0.02
                feedback["message"] = f"Queried metrics: {action.target}"

        elif action_type == "page_team":
            target = action.target.lower() if action.target else ""
            if (
                self._team_should_be_notified
                and target == self.incident.team_to_page.lower()
            ):
                feedback["recognized"] = True
                feedback["points"] = self.SCORE_TEAM_NOTIFICATION
                feedback["message"] = f"Correctly paged {self.incident.team_to_page}"
                self._team_notified = True
            elif self._team_should_be_notified:
                feedback["recognized"] = True
                feedback["message"] = f"Paged team: {target}"
                self._team_notified = True
            else:
                feedback["recognized"] = True
                feedback["message"] = f"Paged team (not needed): {target}"

        elif action_type == "rollback":
            target = action.target.lower() if action.target else ""
            if target == self.incident.affected_service.lower():
                feedback["recognized"] = True
                feedback["points"] = self.SCORE_REMEDIATION
                feedback["message"] = f"Rolled back {self.incident.affected_service}"
                self._remediation_done = True
            elif target:
                feedback["recognized"] = True
                feedback["points"] = 0.02
                feedback["message"] = f"Rolled back {target} (wrong service)"
                self.wrong_actions += 1

        elif action_type == "scale_service":
            target = action.target.lower() if action.target else ""
            if target.startswith(self.incident.affected_service.lower()):
                feedback["recognized"] = True
                feedback["points"] = self.SCORE_REMEDIATION
                feedback["message"] = f"Scaled {self.incident.affected_service}"
                self._remediation_done = True
            elif target:
                feedback["recognized"] = True
                feedback["points"] = 0.02
                feedback["message"] = f"Scaled {target} (possibly wrong)"
                self.wrong_actions += 1

        elif action_type == "restart_service":
            target = action.target.lower() if action.target else ""
            if target == self.incident.affected_service.lower():
                feedback["recognized"] = True
                feedback["points"] = self.SCORE_REMEDIATION
                feedback["message"] = f"Restarted {self.incident.affected_service}"
                self._remediation_done = True
            elif target:
                feedback["recognized"] = True
                feedback["points"] = 0.02
                feedback["message"] = f"Restarted {target}"
                self.wrong_actions += 1

        elif action_type == "toggle_feature":
            target = action.target.lower() if action.target else ""
            if "off" in target:
                feedback["recognized"] = True
                feedback["points"] = self.SCORE_REMEDIATION
                feedback["message"] = f"Disabled feature flag"
                self._remediation_done = True
            elif target:
                feedback["recognized"] = True
                feedback["message"] = f"Toggled feature: {target}"

        elif action_type == "post_update":
            feedback["recognized"] = True
            feedback["points"] = self.SCORE_COMMUNICATION
            feedback["message"] = "Posted status update"
            self._communication_done = True

        elif action_type == "resolve":
            feedback["recognized"] = True
            feedback["points"] = self.SCORE_RESOLUTION
            feedback["message"] = "Incident marked as resolved"
            self._resolution_done = True

        elif action_type == "escalate":
            feedback["recognized"] = True
            feedback["message"] = "Incident escalated"
            self.wrong_actions += 1

        elif action_type == "wait":
            feedback["recognized"] = True
            feedback["message"] = "Waited"
            self.wrong_actions += 0.5

        return feedback

    def compute_score(self) -> GraderResult:
        """Compute the final score for the episode.

        Returns:
            GraderResult with score and breakdown
        """
        base_score = 0.0
        breakdown: Dict[str, Any] = {}

        breakdown["investigation"] = (
            self.SCORE_INVESTIGATION if self._investigation_done else 0.0
        )
        base_score += breakdown["investigation"]

        breakdown["remediation"] = (
            self.SCORE_REMEDIATION if self._remediation_done else 0.0
        )
        base_score += breakdown["remediation"]

        if self._team_should_be_notified:
            breakdown["team_notification"] = (
                self.SCORE_TEAM_NOTIFICATION if self._team_notified else 0.0
            )
        else:
            breakdown["team_notification"] = "N/A"
        base_score += (
            breakdown.get("team_notification", 0.0)
            if isinstance(breakdown.get("team_notification"), float)
            else 0.0
        )

        breakdown["communication"] = (
            self.SCORE_COMMUNICATION if self._communication_done else 0.0
        )
        base_score += breakdown["communication"]

        breakdown["resolution"] = (
            self.SCORE_RESOLUTION if self._resolution_done else 0.0
        )
        base_score += breakdown["resolution"]

        wrong_penalty = min(
            self.wrong_actions * self.WRONG_ACTION_PENALTY,
            self.MAX_WRONG_ACTION_PENALTY,
        )
        breakdown["wrong_action_penalty"] = -wrong_penalty
        base_score -= wrong_penalty

        optimal_steps = len(self.incident.correct_actions)
        efficiency_bonus = max(
            0.0,
            self.MAX_EFFICIENCY_BONUS
            - (self._steps_taken - optimal_steps) * self.EFFICIENCY_DECAY_RATE,
        )
        breakdown["efficiency_bonus"] = round(efficiency_bonus, 3)
        base_score += efficiency_bonus

        final_score = max(0.001, min(0.999, round(base_score, 4)))

        return GraderResult(
            task_id=self.incident.incident_id,
            score=final_score,
            passed=final_score >= 0.6,
            details={
                "breakdown": breakdown,
                "steps_taken": self._steps_taken,
                "wrong_actions": int(self.wrong_actions),
                "investigation_done": self._investigation_done,
                "remediation_done": self._remediation_done,
                "team_notified": self._team_notified,
                "communication_done": self._communication_done,
                "resolution_done": self._resolution_done,
            },
            graded_at=datetime.now(timezone.utc).isoformat(),
        )


def grade_episode(actions: List[SreIncidentAction], incident: Incident) -> GraderResult:
    """Grade a complete episode.

    Args:
        actions: List of actions taken during the episode
        incident: The incident scenario

    Returns:
        GraderResult with final score
    """
    grader = EpisodeGrader(incident)
    for action in actions:
        grader.record_action(action)
        grader.evaluate_action(action)
    return grader.compute_score()
