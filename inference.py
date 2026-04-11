import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from openai import OpenAI

from server.environment import SREEnvironment
from sre_env.constants.prompts import SYSTEM_PROMPT
from sre_env.models import SREAction
from sre_env.utils.parser import extract_json

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_STEPS = 30
TIMEOUT_MINUTES = 5

# Tasks the validator will evaluate: (name, seed)
TASKS = [
    ("pod-restart", 2),            # Easy
    ("db-index-optimisation", 1),  # Medium
    ("dynamic-scaling", 5),        # Hard
    ("bad-deployment-rollback", 0),# Extreme
]


def clamp_score(score):
    """Clamp score to strictly (0, 1) — never exactly 0.0 or 1.0."""
    if score is None:
        return 0.05
    score = float(score)
    return max(0.05, min(0.95, score))


def run_task(client, task_name, seed):
    """Run a single task episode and emit [STEP] blocks."""
    print("[START]")
    print(json.dumps({"task": task_name, "seed": seed}))

    env = SREEnvironment()

    obs = env.reset(seed=seed)
    obs_dict = obs.model_dump()
    done = obs_dict.get("done", False)

    total_reward = 0.0
    step_count = 0
    start_time = time.time()

    task_desc = obs_dict.get("task_description", "N/A")

    while not done and step_count < MAX_STEPS:
        if (time.time() - start_time) / 60 > TIMEOUT_MINUTES:
            break

        step_count += 1

        try:
            history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: {json.dumps(obs_dict)}"},
            ]
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=history, max_tokens=512, temperature=0.1
            )
            raw_content = response.choices[0].message.content.strip()
            action_dict = extract_json(raw_content)
        except Exception as e:
            print("[STEP]")
            print(json.dumps({"task": task_name, "error": str(e), "step": step_count}))
            break

        action = SREAction(**action_dict)
        obs = env.step(action)
        obs_dict = obs.model_dump()

        reward = obs_dict.get("reward", 0.0) or 0.0
        done = obs_dict.get("done", False)
        total_reward += reward

        print("[STEP]")
        print(json.dumps({
            "task": task_name,
            "step": step_count,
            "action": action_dict,
            "observation": {
                "message": obs_dict.get("message", ""),
                "success": obs_dict.get("success", False),
            },
            "reward": clamp_score(reward),
            "total_reward": clamp_score(total_reward),
            "done": done,
        }))

    # Extract and clamp grader score
    score = obs_dict.get("metadata", {}).get("grader_score", None)
    final_msg = obs_dict.get("message", "")
    if "GRADER_SCORE:" in final_msg:
        try:
            score = float(final_msg.split("GRADER_SCORE: ")[1].split("]")[0])
        except Exception:
            pass

    clamped_score = clamp_score(score)
    clamped_reward = clamp_score(total_reward)
    
    print("[END]")
    print(json.dumps({
        "task": task_name,
        "total_steps": step_count,
        "total_reward": clamped_reward,
        "grader_score": clamped_score,
        "score": clamped_score,
    }))

    return {
        "task": task_name,
        "seed": seed,
        "task_description": task_desc,
        "total_steps": step_count,
        "total_reward": clamped_reward,
        "grader_score": clamped_score,
        "score": clamped_score,
    }


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

    results = []
    for task_name, seed in TASKS:
        result = run_task(client, task_name, seed)
        results.append(result)

    with open("agent_trace.json", "w") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "tasks": results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
