import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from server.api.routes import agent, environment
from server.core.deps import env_server

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv()

app = FastAPI(
    title="SRE Incident Response Environment",
    description="OpenEnv RL environment simulating autonomous SRE incident response.",
    version="1.0.0",
)

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


if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)
