from fastapi import FastAPI
from warehouse_env.env_core import WarehouseEnv
from warehouse_env.models import Action

app = FastAPI()
env = WarehouseEnv()


# ✅ Reset (POST required for OpenEnv)
@app.post("/reset")
def reset():
    return env.reset()


# ✅ Step
@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.value,
        "done": done,
        "info": info,
    }


# ✅ State
@app.get("/state")
def state():
    return env.state()


# ✅ OpenEnv entrypoint (IMPORTANT)
def main():
    return app


# ✅ Local run only (safe)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)