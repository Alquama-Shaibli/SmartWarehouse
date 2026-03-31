from fastapi import FastAPI
from warehouse_env.warehouse_env import WarehouseEnv
from warehouse_env.models import Action

app = FastAPI()
env = WarehouseEnv()


@app.post("/reset")
@app.get("/reset")
def reset():
    return env.reset()


@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.value,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state()


def main():
    return app
