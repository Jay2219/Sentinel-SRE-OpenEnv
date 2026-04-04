import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from server.api.routes import environment
from server.core.deps import env_server

app = FastAPI(
    title="SRE Incident Response Environment",
    description="OpenEnv RL environment simulating autonomous SRE incident response.",
    version="1.0.0",
)

# Standard endpoints (OpenEnv framework)
env_server.register_routes(app)

# Custom extensions
app.include_router(environment.router, tags=["SRE Chaos"])


@app.get("/health", tags=["Utilities"])
async def health() -> JSONResponse:
    """Return 200 OK for container health checks."""
    return JSONResponse(content={"status": "ok"}, status_code=200)


if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)
