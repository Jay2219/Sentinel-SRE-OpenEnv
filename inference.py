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
TIMEOUT_MINUTES = 18


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
    env = SREEnvironment()

    # ── [START] ──
    print("[START]")
    print(json.dumps({"agent": "sentinel-sre", "mode": "automated", "seed": 2}))

    obs = env.reset(seed=2)
    obs_dict = obs.model_dump()
    done = obs_dict.get("done", False)

    total_reward = 0.0
    step_count = 0
    start_time = time.time()

    task = obs_dict.get("task_description", "N/A")
    print(json.dumps({"task_description": task}))

    while not done and step_count < MAX_STEPS:
        if (time.time() - start_time) / 60 > TIMEOUT_MINUTES:
            print("[STEP]")
            print(json.dumps({"error": "timeout", "step": step_count}))
            break

        step_count += 1

        # ── [STEP] ──
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
            print(json.dumps({"error": str(e), "step": step_count}))
            break

        action = SREAction(**action_dict)
        obs = env.step(action)
        obs_dict = obs.model_dump()

        reward = obs_dict.get("reward", 0.0) or 0.0
        done = obs_dict.get("done", False)
        total_reward += reward

        print("[STEP]")
        print(json.dumps({
            "step": step_count,
            "action": action_dict,
            "observation": {
                "message": obs_dict.get("message", ""),
                "success": obs_dict.get("success", False),
            },
            "reward": reward,
            "total_reward": total_reward,
            "done": done,
        }))

    # ── [END] ──
    score = obs_dict.get("metadata", {}).get("grader_score", None)
    final_msg = obs_dict.get("message", "")
    if "GRADER_SCORE:" in final_msg:
        try:
            score = float(final_msg.split("GRADER_SCORE: ")[1].split("]")[0])
        except Exception:
            pass

    print("[END]")
    print(json.dumps({
        "total_steps": step_count,
        "total_reward": total_reward,
        "grader_score": score,
    }))

    with open("agent_trace.json", "w") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "task_description": task,
            "total_steps": step_count,
            "grader_score": score,
        }, f, indent=2)


if __name__ == "__main__":
    main()
