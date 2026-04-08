import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests
from dotenv import load_dotenv
from openai import OpenAI

from sre_env.constants.prompts import SYSTEM_PROMPT
from sre_env.core.client import env_reset, env_step
from sre_env.utils.parser import extract_json

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")

MAX_STEPS = 30
TIMEOUT_MINUTES = 18


def main():
    # Non-interactive mode: auto-select Easy (seed=2)
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

    print("🛡️ Sentinel-SRE Autonomous Agent")
    print("Running in automated mode (seed=2, Easy)...\n")

    reset_resp = env_reset(ENV_BASE_URL, seed=2)
    observation = reset_resp.get("observation", {})
    done = reset_resp.get("done", False)

    total_reward = 0.0
    step_count = 0
    start_time = time.time()

    audit_trail = {
        "timestamp": datetime.utcnow().isoformat(),
        "task_description": observation.get("task_description", "N/A"),
        "trajectory": [],
    }

    print(f"📋 Task: {audit_trail['task_description']}")

    while not done and step_count < MAX_STEPS:
        if (time.time() - start_time) / 60 > TIMEOUT_MINUTES:
            print("⏰ Timeout Reached!")
            break

        step_count += 1
        print(f"\n── Step {step_count} ──")

        try:
            history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context: {json.dumps(observation)}"},
            ]
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=history, max_tokens=512, temperature=0.1
            )
            raw_content = response.choices[0].message.content.strip()
            action = extract_json(raw_content)
        except Exception as e:
            print(f"❌ LLM/Parse Error: {e}")
            break

        cmd_type = action.get("command_type", "UNKNOWN")
        target = action.get("target_resource", "UNKNOWN")
        print(f"  🚀 Action: {cmd_type} on {target}")

        step_resp = env_step(ENV_BASE_URL, action)
        prev_obs = observation
        observation = step_resp.get("observation", {})
        reward = step_resp.get("reward", 0.0) or 0.0
        done = step_resp.get("done", False)
        total_reward += reward

        audit_trail["trajectory"].append({
            "step": step_count,
            "observation_in": prev_obs,
            "action": action,
            "reward": reward,
            "observation_out": observation,
        })

        msg = observation.get("message", "")
        print(f"  ► Result: {msg}")
        print(f"  ► Reward: {reward:+.3f} | Total: {total_reward:+.3f}")

    # Extract final score
    score = observation.get("metadata", {}).get("grader_score", "N/A")
    final_msg = observation.get("message", "")
    if "GRADER_SCORE:" in final_msg:
        try:
            score = final_msg.split("GRADER_SCORE: ")[1].split("]")[0]
        except Exception:
            pass

    print(f"\n{'='*40}")
    print(f"Final Grade: {score}")
    print(f"Total Steps: {step_count}")
    print(f"{'='*40}")

    with open("agent_trace.json", "w") as f:
        json.dump(audit_trail, f, indent=2)
    print("Audit trail saved to agent_trace.json")


if __name__ == "__main__":
    main()
