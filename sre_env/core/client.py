import sys

import requests


def env_reset(base_url: str, seed: int | None = None) -> dict:
    try:
        payload = {"seed": seed} if seed is not None else {}
        resp = requests.post(f"{base_url}/reset", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Could not connect to environment server at {base_url}")
        sys.exit(1)


def env_step(base_url: str, action: dict) -> dict:
    resp = requests.post(f"{base_url}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json()
