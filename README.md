---
title: SRE Incident Response Environment
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - sre
  - incident-response
  - rl-training
---

# SRE Incident Response Environment

An OpenEnv-compliant environment for training RL agents on Site Reliability Engineering (SRE) incident response tasks.

## Overview

This environment simulates real-world SRE incidents where an RL agent must:

1. **Triage** - Analyze active alerts and understand the scope
2. **Investigate** - Check dashboards, run queries, find root cause
3. **Remediate** - Apply fixes (rollback, scale, restart, toggle features)
4. **Coordinate** - Page appropriate teams when needed
5. **Communicate** - Post status updates
6. **Resolve** - Close the incident properly

## Task Difficulty Levels

| Level | Description | Max Steps | Incident Types |
|-------|-------------|-----------|---------------|
| **Easy** | API/service errors | 15 | Rollback or scale issues (5 templates) |
| **Medium** | Database/infrastructure | 20 | Pool exhaustion, slow queries (5 templates) |
| **Hard** | Complex infrastructure | 25 | CDN, feature flags, DNS (5 templates) |

## Action Space

| Action | Description | Example Target |
|--------|-------------|---------------|
| `list_alerts` | Get active alerts | (empty) |
| `check_dashboard` | View metrics dashboard | `payment-health` |
| `run_query` | Query metrics/logs | `error rate` |
| `get_deployment` | Check recent deploys | `payment-api` |
| `rollback` | Roll back service | `payment-api` |
| `scale_service` | Scale replicas | `pgbouncer:8` |
| `restart_service` | Restart service | `analytics-worker` |
| `toggle_feature` | Toggle feature flag | `cdn_routing:off` |
| `page_team` | Page on-call team | `db-team` |
| `post_update` | Post status update | `status message` |
| `resolve` | Close incident | `resolution summary` |
| `escalate` | Escalate to management | - |
| `wait` | Wait for more data | (empty) |

## Observation Space

```python
SreIncidentObservation:
    incident_id: str       # Unique incident identifier
    severity: str          # P1-P5 severity
    title: str             # Incident title
    description: str       # Incident description
    action_result: str     # Result of last action
    active_alerts: List[str]      # Currently firing alerts
    system_status: Dict[str, str] # Component health
    timeline: List[str]    # Action history
    step: int             # Current step number
    hint: str             # Optional hint
    done: bool            # Episode complete
    reward: float          # Step reward
```

## Scoring

Episode score is calculated from:

| Component | Points |
|-----------|--------|
| Investigation | 0.25 |
| Remediation | 0.35 |
| Team Notification | 0.15 |
| Communication | 0.10 |
| Resolution | 0.15 |
| Efficiency Bonus | up to 0.10 |

**Success Threshold:** 0.60 (60%)

## Quick Start

```python
from sre_incident_env.client import SreIncidentEnv
from sre_incident_env.models import ActionType, SreIncidentAction

# Connect to environment
with SreIncidentEnv(base_url="http://localhost:7860") as env:
    # Start episode
    result = env.reset(task_id="easy")
    print(result.observation.title)
    
    # Take actions
    result = env.step(SreIncidentAction(
        action_type=ActionType.LIST_ALERTS,
        target="",
        reasoning="Check active alerts"
    ))
```

## Running Inference

```bash
# Set environment variables
export HF_TOKEN=your_token_here
export MODEL_NAME=llama-3.1-8b-instant

# Run inference
python inference.py

# Run specific tasks
SRE_TASKS=easy,medium,hard python inference.py
```

Expected output format:
```
[START] task=easy env=sre_incident_env model=llama-3.1-8b-instant
[STEP] step=1 action=list_alerts(...) reward=0.00 done=false
...
[END] success=true steps=6 score=0.850 rewards=0.00,0.10,0.15,...
```

## Baseline Scores

| Task | Expected Score | Description |
|------|----------------|-------------|
| Easy | ~0.65 | Simple rollback/scale issues |
| Medium | ~0.55 | Requires investigation + paging |
| Hard | ~0.45 | Complex multi-step resolution |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/tasks` | GET | List available tasks |
| `/reset` | POST | Reset environment |
| `/step` | POST | Execute action |
| `/state` | GET | Get current state |
| `/grader` | POST | Grade episode |

## Grader Endpoint

The `/grader` endpoint runs a deterministic episode using optimal actions:

```bash
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'
```

Response:
```json
{
  "task_id": "easy",
  "score": 0.85,
  "passed": true,
  "details": {...},
  "graded_at": "2024-01-01T00:00:00Z"
}
```

## Installation

```bash
# Clone the OpenEnv repository
git clone https://github.com/meta-pytorch/OpenEnv.git
cd OpenEnv

# Install dependencies
uv sync

# Run locally
cd envs/sre_incident_env
uv run server
```

## Docker Deployment

```bash
# Build container
docker build -t sre-incident-env:latest -f server/Dockerfile .

# Run container
docker run -p 7860:7860 sre-incident-env:latest
```

## Hugging Face Space

Deploy to HF Spaces:

```bash
huggingface-cli login
uv run openenv push --repo-id YOUR_USERNAME/sre-incident-env
```

Live at: `https://YOUR_USERNAME-sre-incident-env.hf.space`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://api.groq.com/openai/v1` | LLM endpoint |
| `MODEL_NAME` | `llama-3.1-8b-instant` | Model to use |
| `HF_TOKEN` | - | API token |
| `ENV_BASE_URL` | `http://localhost:7860` | Environment URL |
| `SRE_TASK` | `easy` | Task to run |
| `SRE_TASKS` | `easy` | Comma-separated tasks |
| `MAX_STEPS` | `25` | Max steps per episode |

## Project Structure

```
sre_incident_env/
├── __init__.py
├── models.py              # Action, Observation, TaskConfig, GraderResult
├── client.py             # HTTP/WebSocket client
├── inference.py          # OpenAI API inference script
├── openenv.yaml         # OpenEnv specification
├── pyproject.toml       # Package config
├── README.md            # This file
├── requirements.txt
├── uv.lock
└── server/
    ├── __init__.py
    ├── Dockerfile
    ├── app.py                    # FastAPI server
    ├── sre_incident_env_environment.py  # Core environment
    ├── scenario_generator.py     # 15 incident templates
    ├── tasks.py                  # Task definitions
    └── grading.py                # Scoring logic
```

## License

BSD-style license (see LICENSE file)
