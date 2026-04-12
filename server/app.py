import json
import os
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from server.api.routes import agent, environment
from server.core.deps import env_server

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv()

app = FastAPI(
    title="SRE Incident Response Environment",
    description="OpenEnv RL environment simulating autonomous SRE incident response.",
    version="1.0.0",
)


class JSONBoundaryMiddleware(BaseHTTPMiddleware):
    """Nuclear Safety Middleware: Recursively clamps ALL outbound floats to [0.15, 0.85]."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Only intercept JSON responses
        if "application/json" in response.headers.get("Content-Type", ""):
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            try:
                data = json.loads(body)
                safe_data = self._clamp_recursive(data)
                safe_body = json.dumps(safe_data).encode("utf-8")
                
                # Create a new response with safe body
                # We use a custom Response to avoid re-triggering middleware in some versions
                return Response(
                    content=safe_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type="application/json"
                )
            except Exception:
                # If parsing fails, return original body
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type="application/json"
                )
        
        return response

    def _clamp_recursive(self, obj: Any) -> Any:
        if isinstance(obj, float):
            # The Nuclear Clamp: No value survives outside (0.1, 0.9)
            return max(0.15, min(0.85, obj))
        elif isinstance(obj, dict):
            return {k: self._clamp_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clamp_recursive(i) for i in obj]
        return obj


# Register Nuclear Middleware
app.add_middleware(JSONBoundaryMiddleware)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root(request: Request):
    """Aesthetic landing page for the Hugging Face Space."""
    return templates.TemplateResponse(request=request, name="index.html")


# OpenEnv standard endpoints
env_server.register_routes(app)

# SRE Chaos extensions
app.include_router(environment.router, tags=["SRE Chaos"])

# Live agent runner (SSE)
app.include_router(agent.router, tags=["Agent"])


@app.get("/health", tags=["Utilities"])
async def health() -> JSONResponse:
    """Return 200 OK for container health checks."""
    return JSONResponse(content={"status": "ok"}, status_code=200)


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
