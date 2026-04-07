from fastapi import FastAPI
from warehouse_env.env_core import WarehouseEnv
from warehouse_env.models import Action

app = FastAPI(title="Smart Warehouse Environment")

# Initialize environment
env = WarehouseEnv()


# -----------------------------
# HEALTH CHECK (IMPORTANT for HF)
# -----------------------------
@app.get("/")
def root():
    return {"status": "running"}


# -----------------------------
# RESET (OpenEnv expects POST)
# -----------------------------
@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()


# -----------------------------
# STEP
# -----------------------------
@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.dict(),
        "reward": reward.value,
        "done": done,
        "info": info,
    }


# -----------------------------
# STATE
# -----------------------------
@app.get("/state")
def state():
    return env.state()


# -----------------------------
# OPENENV ENTRYPOINT (CRITICAL)
# -----------------------------
def main():
    return app


# -----------------------------
# LOCAL RUN ONLY
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)