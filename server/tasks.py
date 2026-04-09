# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task definitions for SRE Incident Response Environment.

Defines the task catalog with 3 difficulty levels:
- Easy: API/service errors requiring simple rollback or scaling
- Medium: Database/infrastructure issues requiring investigation and team coordination
- Hard: Complex infrastructure problems requiring feature flags and cross-team response
"""

from typing import Dict, List

from ..models import TaskConfig

TASKS: Dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        task_name="API/Service Error Resolution",
        difficulty="easy",
        description=(
            "Resolve API or service-level errors (HTTP 500/502/503). "
            "Typical issues include deployment failures, configuration errors, "
            "or resource exhaustion that can be resolved with rollback or scaling."
        ),
        metric="score",
        max_steps=15,
        success_threshold=0.6,
    ),
    "medium": TaskConfig(
        task_id="medium",
        task_name="Database & Infrastructure Investigation",
        difficulty="medium",
        description=(
            "Investigate and resolve database or infrastructure issues. "
            "Requires querying metrics, identifying root cause, coordinating "
            "with the appropriate team, and applying the fix."
        ),
        metric="score",
        max_steps=20,
        success_threshold=0.6,
    ),
    "hard": TaskConfig(
        task_id="hard",
        task_name="Complex Infrastructure & Feature Flag Issues",
        difficulty="hard",
        description=(
            "Resolve complex infrastructure issues involving CDN, load balancers, "
            "feature flags, DNS, or TLS. Requires careful investigation, "
            "cross-team coordination, and precise configuration changes."
        ),
        metric="score",
        max_steps=25,
        success_threshold=0.6,
    ),
}


def get_task(task_id: str) -> TaskConfig:
    """Get a task configuration by ID.

    Args:
        task_id: The task identifier (easy/medium/hard)

    Returns:
        TaskConfig for the requested task

    Raises:
        ValueError: If task_id is not found
    """
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def get_all_tasks() -> List[TaskConfig]:
    """Get all available tasks.

    Returns:
        List of all TaskConfig objects
    """
    return list(TASKS.values())


def get_task_ids() -> List[str]:
    """Get all task IDs.

    Returns:
        List of task identifiers
    """
    return list(TASKS.keys())
