import json
import os
import sys
import time
from datetime import datetime

import requests
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sre_env.constants.prompts import ROUTER_SYSTEM_PROMPT, SYSTEM_PROMPT
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
    console = Console()
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

    console.print(
        Panel.fit(
            "[bold cyan]🛡️ Sentinel-SRE Autonomous Agent[/bold cyan]\nInteractive Incident Injector",
            border_style="cyan",
        )
    )

    console.print("\n[bold white]Select an Incident to inject into the environment:[/bold white]")
    console.print("  [green][1][/green] 🟢 OOMKilled Pod Crash (Easy)")
    console.print("  [yellow][2][/yellow] 🟡 Missing Database Index (Medium)")
    console.print("  [red][3][/red] 🔴 50,000 RPS Traffic Spike (Hard)")
    console.print("  [magenta][4][/magenta] 💀 Bad Code Deployment (Extreme)")
    console.print("  [blue][5][/blue] 🧠 Dynamic Task Generation (Custom Prompt)")

    while True:
        choice = Prompt.ask(
            "\n[bold cyan]Enter scenario[/bold cyan]",
            choices=["1", "2", "3", "4", "5"],
            default="1",
        )

        if choice == "5":
            user_prompt = Prompt.ask("\n[bold blue]Describe the incident you want to simulate[/bold blue]")

            with console.status("[bold blue]LLM Analyzing Scenario...", spinner="dots"):
                try:
                    resp = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format={"type": "json_object"},
                    )
                    result = json.loads(resp.choices[0].message.content)
                    if not result.get("accepted"):
                        console.print(f"[bold red]❌ Rejected:[/bold red] {result.get('reason')}")
                        continue

                    diff_map = {"easy": 2, "medium": 1, "hard": 5, "extreme": 0}
                    chosen_seed = diff_map.get(result.get("archetype", "easy"), 2)

                    with console.status("[bold green]Injecting Custom Chaos...", spinner="aesthetic"):
                        payload = {
                            "seed": chosen_seed,
                            "difficulty": result.get("archetype", "easy"),
                            "custom_description": result["custom_description"],
                            "custom_logs": result.get("custom_logs", []),
                        }
                        r = requests.post(f"{ENV_BASE_URL}/custom_reset", json=payload, timeout=10)
                        reset_resp = r.json()
                        break
                except Exception as e:
                    console.print(f"[bold red]LLM Error:[/bold red] {e}")
                    continue
        else:
            seed_map = {"1": 2, "2": 1, "3": 5, "4": 0}
            chosen_seed = seed_map[choice]
            with console.status("[bold green]Injecting Static Chaos...", spinner="aesthetic"):
                reset_resp = env_reset(ENV_BASE_URL, seed=chosen_seed)
            break

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

    console.print(f"[bold yellow]📋 Task:[/bold yellow] {audit_trail['task_description']}")

    while not done and step_count < MAX_STEPS:
        if (time.time() - start_time) / 60 > TIMEOUT_MINUTES:
            console.print("[bold red]Timeout Reached![/bold red]")
            break

        step_count += 1
        console.print(f"\n[bold magenta]── Step {step_count} ──────────────────────────────────[/bold magenta]")

        action = {}
        with console.status("[bold green]Agent Thinking...", spinner="dots"):
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
                console.print(f"[bold red]❌ LLM/Parse Error:[/bold red] {e}")
                break

        cmd_type = action.get("command_type", "UNKNOWN")
        target = action.get("target_resource", "UNKNOWN")
        console.print(f"  [bold blue]🚀 Action Request:[/bold blue] {cmd_type} [white]on[/white] {target}")

        step_resp = env_step(ENV_BASE_URL, action)
        prev_obs = observation
        observation = step_resp.get("observation", {})
        reward = step_resp.get("reward", 0.0) or 0.0
        done = step_resp.get("done", False)
        total_reward += reward

        audit_trail["trajectory"].append(
            {
                "step": step_count,
                "observation_in": prev_obs,
                "action": action,
                "reward": reward,
                "observation_out": observation,
            }
        )

        msg = observation.get("message", "")
        color = "green" if "Success" in msg or "restored" in msg else "yellow" if "Already" in msg else "red"
        console.print(f"  [bold {color}]► Result:[/bold {color}] {msg}")
        console.print(
            f"  [bold cyan]► Reward:[/bold cyan] {reward:+.3f} | [bold cyan]Total:[/bold cyan] {total_reward:+.3f}"
        )

    score = observation.get("metadata", {}).get("grader_score", "N/A")
    final_msg = observation.get("message", "")
    if "GRADER_SCORE:" in final_msg:
        try:
            score = final_msg.split("GRADER_SCORE: ")[1].split("]")[0]
        except Exception:
            pass

    panel = Panel.fit(
        f"[bold white]Resolution Complete[/bold white]\n"
        f"[bold green]Final Grade:[/bold green] {score}\n"
        f"[bold green]Total Steps:[/bold green] {step_count}",
        title="[bold cyan]End of Episode[/bold cyan]",
        border_style="cyan",
    )
    console.print("\n", panel)

    with open("agent_trace.json", "w") as f:
        json.dump(audit_trail, f, indent=2)
    console.print("[dim]Audit trail saved to agent_trace.json[/dim]")


if __name__ == "__main__":
    main()
