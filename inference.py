#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference Script for SRE Incident Response Environment.

Uses the OpenAI API client to run a model against the SRE incident
response environment and produces reproducible baseline scores.

STDOUT FORMAT (mandatory):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<r> done=<true|false>
    [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables:
    API_BASE_URL    LLM endpoint (default: https://api.groq.com/openai/v1)
    MODEL_NAME      Model identifier (default: llama-3.1-8b-instant)
    HF_TOKEN        Hugging Face API token (or OPENAI_API_KEY)
    ENV_BASE_URL    HF Space base URL (default: https://saumy-a-sre-incident-env.hf.space)
    SRE_TASK        Task to run: easy | medium | hard (default: easy)
    SRE_TASKS       Comma-separated tasks: easy,medium,hard (default: easy)
    MAX_STEPS       Maximum steps per episode (default: 25)
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import SreIncidentEnv
from models import ActionType, SreIncidentAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "sre_incident_env"
MAX_STEPS = int(os.getenv("MAX_STEPS", "25"))
SUCCESS_THRESHOLD = 0.6
TEMPERATURE = 0.0
MAX_TOKENS = 400


def log_start(task: str, env: str, model: str) -> None:
    """Log episode start.

    Args:
        task: Task identifier
        env: Benchmark name
        model: Model name
    """
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str] = None
) -> None:
    """Log step execution.

    Args:
        step: Step number
        action: Action taken
        reward: Reward received
        done: Whether episode is done
        error: Optional error message
    """
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end.

    Args:
        success: Whether episode succeeded
        steps: Total steps taken
        score: Final score
        rewards: List of all rewards
    """
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Site Reliability Engineer (SRE) responding to a production incident.

    At each step respond with a JSON object — no markdown, no explanation outside the JSON:
    {
      "action_type": "<one of: list_alerts, check_dashboard, run_query, get_deployment, rollback, scale_service, restart_service, toggle_feature, page_team, post_update, resolve, escalate, wait>",
      "target": "<target of the action (service name, query, etc.)>",
      "reasoning": "<brief explanation>"
    }

    Action guide:
    - list_alerts       → get all firing alerts (target: empty string)
    - check_dashboard   → view dashboard (target: dashboard name)
    - run_query         → metric/log query (target: query string)
    - get_deployment    → recent deploys (target: service name)
    - rollback          → roll back service (target: service name)
    - scale_service     → scale up/down (target: 'service:replicas')
    - restart_service   → restart service (target: service name)
    - toggle_feature    → enable/disable flag (target: 'flag_name:on|off')
    - page_team         → page on-call team (target: team name)
    - post_update       → status comms (target: update message)
    - resolve           → close incident (target: resolution summary)

    Strategy:
    1. Start by listing alerts to understand scope
    2. Check dashboards and run queries to find root cause
    3. Apply the correct remediation (rollback, scale, restart, or toggle)
    4. Page the right team if specialized help is needed
    5. Post a status update
    6. Resolve the incident once fixed
""").strip()


def obs_to_prompt(obs) -> str:
    """Convert observation to user prompt for LLM.

    Args:
        obs: SreIncidentObservation

    Returns:
        Formatted prompt string
    """
    alerts = "\n".join(f"  - {a}" for a in (obs.active_alerts or [])) or "  (none)"
    status = (
        "\n".join(f"  {k}: {v}" for k, v in (obs.system_status or {}).items())
        or "  (unknown)"
    )
    timeline = (
        "\n".join(f"  {t}" for t in (obs.timeline or [])[-5:]) or "  (no actions yet)"
    )

    msg = (
        f"=== INCIDENT {obs.incident_id} | Severity: {obs.severity} ===\n"
        f"Title: {obs.title}\n\n"
        f"Description:\n{obs.description}\n\n"
        f"LAST ACTION RESULT:\n{obs.action_result}\n\n"
        f"ACTIVE ALERTS:\n{alerts}\n\n"
        f"SYSTEM STATUS:\n{status}\n\n"
        f"RECENT TIMELINE (last 5):\n{timeline}\n\n"
        f"Step: {obs.step}"
    )

    if obs.hint:
        msg += f"\n\nHINT: {obs.hint}"

    return msg


def parse_action(content: str) -> SreIncidentAction:
    """Parse LLM response into SreIncidentAction.

    Args:
        content: Raw LLM response

    Returns:
        SreIncidentAction instance
    """
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
            if content.startswith("json"):
                content = content[4:]
    content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        content_clean = content.strip("{}").strip()
        data = {}
        for line in content_clean.split(","):
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip().strip('"').strip("'")] = (
                    value.strip().strip(",").strip('"').strip("'")
                )

    return SreIncidentAction(
        action_type=ActionType(data.get("action_type", "wait")),
        target=data.get("target", ""),
        reasoning=data.get("reasoning", ""),
    )


async def run_episode(task_id: str, client) -> dict:
    """Run a single episode for the given task.

    Args:
        task_id: Task identifier (easy/medium/hard)
        client: OpenAI client

    Returns:
        Episode results dictionary
    """
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        result = await client.reset(task_id=task_id)
        obs = result.observation if hasattr(result, "observation") else result

        for step in range(1, MAX_STEPS + 1):
            done = getattr(obs, "done", False) or getattr(result, "done", False)
            if done:
                break

            user_msg = obs_to_prompt(obs)
            messages.append({"role": "user", "content": user_msg})

            try:
                try:
                    from openai import OpenAI

                    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
                    completion = llm_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    content = (completion.choices[0].message.content or "").strip()
                except ImportError:
                    import httpx

                    response = httpx.post(
                        f"{API_BASE_URL}/chat/completions",
                        json={
                            "model": MODEL_NAME,
                            "messages": messages,
                            "temperature": TEMPERATURE,
                            "max_tokens": MAX_TOKENS,
                        },
                        headers={"Authorization": f"Bearer {API_KEY}"},
                        timeout=60.0,
                    )
                    response.raise_for_status()
                    data = response.json()
                    content = (
                        data.get("choices", [{}])[0].get("message", {}).get("content")
                        or ""
                    ).strip()

                messages.append({"role": "assistant", "content": content})
            except Exception as exc:
                content = (
                    '{"action_type": "wait", "target": "", "reasoning": "LLM error"}'
                )
                messages.append({"role": "assistant", "content": content})
                print(f"[DEBUG] LLM error step {step}: {exc}", flush=True)

            error_msg = None
            try:
                action = parse_action(content)
            except Exception as exc:
                error_msg = str(exc)[:80]
                action = SreIncidentAction(action_type=ActionType.WAIT, target="")

            result = await client.step(action)
            obs = result.observation if hasattr(result, "observation") else result
            reward = getattr(result, "reward", 0.0) or 0.0
            done = getattr(result, "done", False)

            rewards.append(reward)
            steps_taken = step

            action_type_val = (
                action.action_type.value
                if hasattr(action.action_type, "value")
                else str(action.action_type)
            )
            action_str = f"{action_type_val}(target={action.target[:40]!r})"
            log_step(
                step=step, action=action_str, reward=reward, done=done, error=error_msg
            )

            if done:
                break

        raw_score = sum(rewards)
        score = min(max(raw_score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        try:
            await client.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_id, "score": score, "success": success, "steps": steps_taken}


async def main() -> None:
    """Main entry point for inference."""
    if not API_KEY:
        raise RuntimeError(
            "No API key found. Set HF_TOKEN, OPENAI_API_KEY, or API_KEY environment variable."
        )

    tasks_str = os.getenv("SRE_TASKS", os.getenv("SRE_TASK", "easy"))
    tasks = [t.strip() for t in tasks_str.split(",")]

    print(f"[INFO] Starting inference with model={MODEL_NAME}", flush=True)
    print(f"[INFO] Tasks: {tasks}", flush=True)
    print(f"[INFO] Environment: {ENV_BASE_URL}", flush=True)
    print("-" * 60, flush=True)

    for task_id in tasks:
        async with SreIncidentEnv(base_url=ENV_BASE_URL) as client:
            await run_episode(task_id, client)
        print("-" * 60, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
