# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scenario Generator for SRE Incident Response Environment.

Generates varied incidents dynamically based on difficulty level and templates.
Each difficulty level has 5 different incident templates that are randomly selected.
"""

import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Incident:
    """Represents an SRE incident scenario."""

    incident_id: str
    title: str
    description: str
    severity: str
    affected_service: str
    symptoms: List[str]
    root_cause: str
    correct_actions: List[Dict[str, str]]
    team_to_page: str
    dashboard_name: str
    query_hint: str


EASY_TEMPLATES = [
    {
        "title": "Payment API 500 errors after deployment",
        "affected_service": "payment-api",
        "symptoms": [
            "Payment API returning HTTP 500 errors",
            "Error rate spiked from 0.1% to 45%",
            "Payment success rate dropped to 55%",
        ],
        "root_cause": "Recent deployment introduced a breaking change",
        "correct_action": {"type": "ROLLBACK", "target": "payment-api"},
        "team_to_page": None,
        "dashboard_name": "payment-health",
        "query_hint": "payment-api error rate",
    },
    {
        "title": "Auth Service timeout after config change",
        "affected_service": "auth-service",
        "symptoms": [
            "Auth Service latency increased to 8s avg",
            "Login timeout errors increasing",
            "Session establishment failures",
        ],
        "root_cause": "Configuration change to session timeout was incorrect",
        "correct_action": {"type": "ROLLBACK", "target": "auth-service"},
        "team_to_page": None,
        "dashboard_name": "auth-metrics",
        "query_hint": "auth timeout rate",
    },
    {
        "title": "Gateway 503 errors from resource exhaustion",
        "affected_service": "api-gateway",
        "symptoms": [
            "API Gateway returning HTTP 503",
            "Connection pool at 100% capacity",
            "Upstream service timeouts",
        ],
        "root_cause": "Connection pool settings were too low for traffic",
        "correct_action": {"type": "SCALE_SERVICE", "target": "api-gateway:10"},
        "team_to_page": None,
        "dashboard_name": "gateway-health",
        "query_hint": "gateway connections",
    },
    {
        "title": "User Service 502 from upstream timeout",
        "affected_service": "user-service",
        "symptoms": [
            "User Service returning HTTP 502",
            "Profile API failures",
            "User data retrieval errors",
        ],
        "root_cause": "Database connection pool exhausted",
        "correct_action": {"type": "SCALE_SERVICE", "target": "user-service:8"},
        "team_to_page": None,
        "dashboard_name": "user-service-health",
        "query_hint": "user service upstream errors",
    },
    {
        "title": "Order API 500 from database connection issue",
        "affected_service": "order-api",
        "symptoms": [
            "Order API HTTP 500 errors",
            "Checkout failures increasing",
            "Database connection errors",
        ],
        "root_cause": "Order service deployment had connection leak",
        "correct_action": {"type": "ROLLBACK", "target": "order-api"},
        "team_to_page": None,
        "dashboard_name": "order-metrics",
        "query_hint": "order-api database connections",
    },
]

MEDIUM_TEMPLATES = [
    {
        "title": "Database connection pool exhaustion causing 503s",
        "affected_service": "pgbouncer",
        "symptoms": [
            "Multiple services returning HTTP 503",
            "Database connection pool at maximum",
            "New connections being rejected",
            "Error: connection pool exhausted",
        ],
        "root_cause": "pgbouncer connection pool too small for current load",
        "correct_action": {"type": "SCALE_SERVICE", "target": "pgbouncer:8"},
        "team_to_page": "db-team",
        "dashboard_name": "database-health",
        "query_hint": "pgbouncer connection pool",
    },
    {
        "title": "Slow queries causing latency spike",
        "affected_service": "postgres-primary",
        "symptoms": [
            "API latency increased to 3s",
            "Slow query count up 500%",
            "Some queries timing out",
        ],
        "root_cause": "Missing index on orders table after schema change",
        "correct_action": {"type": "RUN_QUERY", "target": "identify slow queries"},
        "team_to_page": "db-team",
        "dashboard_name": "database-health",
        "query_hint": "slow queries",
    },
    {
        "title": "Replication lag causing read inconsistencies",
        "affected_service": "mysql-replica",
        "symptoms": [
            "Stale data in read replicas",
            "Users seeing inconsistent results",
            "Replication lag over 30 seconds",
        ],
        "root_cause": "Replica cannot keep up with primary write rate",
        "correct_action": {"type": "RUN_QUERY", "target": "replication lag"},
        "team_to_page": "db-team",
        "dashboard_name": "database-health",
        "query_hint": "replication lag seconds",
    },
    {
        "title": "Memory leak causing OOM crashes",
        "affected_service": "analytics-worker",
        "symptoms": [
            "Analytics Worker pods restarting",
            "OOMKilled status in kubernetes",
            "Memory usage climbing steadily",
        ],
        "root_cause": "Memory leak in analytics processing",
        "correct_action": {"type": "RESTART_SERVICE", "target": "analytics-worker"},
        "team_to_page": "platform-team",
        "dashboard_name": "analytics-health",
        "query_hint": "analytics memory usage",
    },
    {
        "title": "Cache stampede causing service degradation",
        "affected_service": "redis-cache",
        "symptoms": [
            "Cache hit ratio dropped to 20%",
            "Backend services overloaded",
            "High database load from cache misses",
        ],
        "root_cause": "Cache invalidation storm after deployment",
        "correct_action": {"type": "WAIT", "target": ""},
        "team_to_page": None,
        "dashboard_name": "cache-health",
        "query_hint": "cache hit ratio",
    },
]

HARD_TEMPLATES = [
    {
        "title": "CDN misconfiguration causing global latency",
        "affected_service": "cdn-edge",
        "symptoms": [
            "Global latency increased 300ms",
            "CDN cache miss rate at 80%",
            "Origin server load doubled",
        ],
        "root_cause": "CDN routing rules changed to bypass cache",
        "correct_action": {
            "type": "TOGGLE_FEATURE",
            "target": "cdn_aggressive_routing:off",
        },
        "team_to_page": "cdn-team",
        "dashboard_name": "cdn-edge-health",
        "query_hint": "cdn cache routing",
    },
    {
        "title": "Load balancer misconfiguration causing 50% errors",
        "affected_service": "nginx-lb",
        "symptoms": [
            "50% of requests returning 502",
            "Upstream marked as down incorrectly",
            "Health checks failing intermittently",
        ],
        "root_cause": "Load balancer health check misconfigured",
        "correct_action": {"type": "TOGGLE_FEATURE", "target": "lb_strict_health:off"},
        "team_to_page": "infra-team",
        "dashboard_name": "lb-health",
        "query_hint": "lb upstream errors",
    },
    {
        "title": "Feature flag bug causing cascading failures",
        "affected_service": "recommendation-engine",
        "symptoms": [
            "Recommendation API returning empty results",
            "Feature flag evaluation errors",
            "Cascading timeouts downstream",
        ],
        "root_cause": "New feature flag evaluated to wrong value",
        "correct_action": {
            "type": "TOGGLE_FEATURE",
            "target": "new_recommendation_v2:off",
        },
        "team_to_page": "product-team",
        "dashboard_name": "recommendation-health",
        "query_hint": "feature flag errors",
    },
    {
        "title": "DNS resolution failure affecting multiple services",
        "affected_service": "core-dns",
        "symptoms": [
            "Intermittent DNS resolution failures",
            "Services unable to discover each other",
            "Connection failures across services",
        ],
        "root_cause": "DNS cache poisoning from stale records",
        "correct_action": {"type": "RUN_QUERY", "target": "dns resolution failures"},
        "team_to_page": "infra-team",
        "dashboard_name": "dns-health",
        "query_hint": "dns lookup failures",
    },
    {
        "title": "Certificate expiry causing TLS handshake failures",
        "affected_service": "api-gateway-tls",
        "symptoms": [
            "TLS handshake failures",
            "Certificate expiry warnings",
            "Some clients unable to connect",
        ],
        "root_cause": "TLS certificate expired for api-gateway",
        "correct_action": {"type": "PAGE_TEAM", "target": "security-team"},
        "team_to_page": "security-team",
        "dashboard_name": "tls-cert-health",
        "query_hint": "certificate expiry",
    },
]


class ScenarioGenerator:
    """Generates random SRE incidents based on difficulty level."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize the scenario generator.

        Args:
            seed: Optional random seed for reproducibility
        """
        self._rng = random.Random(seed)

    def generate(self, difficulty: str, seed: Optional[int] = None) -> Incident:
        """Generate a random incident for the given difficulty.

        Args:
            difficulty: Difficulty level (easy/medium/hard)
            seed: Optional seed override for this generation

        Returns:
            Incident object with all scenario details
        """
        rng = random.Random(seed)

        if difficulty == "easy":
            template = rng.choice(EASY_TEMPLATES)
        elif difficulty == "medium":
            template = rng.choice(MEDIUM_TEMPLATES)
        elif difficulty == "hard":
            template = rng.choice(HARD_TEMPLATES)
        else:
            template = rng.choice(EASY_TEMPLATES)

        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        severity = "P1" if difficulty == "hard" else "P2"

        description = f"""
{", ".join(template["symptoms"][:2])}

Affected Service: {template["affected_service"]}

Recent Context: {template["root_cause"]}
        """.strip()

        correct_actions = [
            {"action_type": "LIST_ALERTS", "target": ""},
            {"action_type": "CHECK_DASHBOARD", "target": template["dashboard_name"]},
            {"action_type": "RUN_QUERY", "target": template["query_hint"]},
        ]

        if template.get("team_to_page"):
            correct_actions.append(
                {"action_type": "PAGE_TEAM", "target": template["team_to_page"]}
            )

        correct_actions.append(template["correct_action"])
        correct_actions.append(
            {"action_type": "POST_UPDATE", "target": "Status update posted"}
        )
        correct_actions.append(
            {"action_type": "RESOLVE", "target": "Incident resolved"}
        )

        return Incident(
            incident_id=incident_id,
            title=template["title"],
            description=description,
            severity=severity,
            affected_service=template["affected_service"],
            symptoms=template["symptoms"],
            root_cause=template["root_cause"],
            correct_actions=correct_actions,
            team_to_page=template.get("team_to_page"),
            dashboard_name=template["dashboard_name"],
            query_hint=template["query_hint"],
        )

    def get_all_templates(self, difficulty: str) -> List[Dict[str, Any]]:
        """Get all templates for a difficulty level.

        Args:
            difficulty: Difficulty level

        Returns:
            List of template dictionaries
        """
        if difficulty == "easy":
            return EASY_TEMPLATES
        elif difficulty == "medium":
            return MEDIUM_TEMPLATES
        elif difficulty == "hard":
            return HARD_TEMPLATES
        return []
