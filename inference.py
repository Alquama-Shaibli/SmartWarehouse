import os
import sys

# ── Environment variables exactly as required by the sample ──────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "<your-active-api-base-url>")
MODEL_NAME       = os.getenv("MODEL_NAME",   "<your-active-model-name>")
HF_TOKEN         = os.getenv("HF_TOKEN")                  # no default
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")           # optional

# ── OpenAI client configured via the injected env variables ──────────────────
from openai import OpenAI

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy-key",   # HF_TOKEN is the auth key for the proxy
)

# ── Env import with graceful fallback ─────────────────────────────────────────
try:
    from warehouse_env.env_core import WarehouseEnv
    from warehouse_env.models import Action
except ImportError:
    try:
        from env_core import WarehouseEnv
        from models import Action
    except ImportError:
        WarehouseEnv = None
        Action = None

# ── Tasks & constants ─────────────────────────────────────────────────────────
TASKS     = ["easy", "medium", "hard"]
MAX_STEPS = 50


# ── Heuristic fallback agent ──────────────────────────────────────────────────
def get_heuristic_action(obs, env):
    sm = env.state_manager
    rx, ry = obs.robot_position
    gx, gy = obs.goal
    cx, cy = sm.charge_station

    if obs.battery < 20 and sm.robot != [cx, cy]:
        if rx > cx:   return Action(action_type="move", direction="up")
        elif rx < cx: return Action(action_type="move", direction="down")
        elif ry > cy: return Action(action_type="move", direction="left")
        else:         return Action(action_type="move", direction="right")

    elif len(sm.carrying) < 3:
        target = None
        for item, pos in sm.inventory.items():
            if item not in sm.carrying:
                target = pos
                break
        if target:
            ix, iy = target
            if rx == ix and ry == iy: return Action(action_type="pick")
            elif rx < ix:  return Action(action_type="move", direction="down")
            elif rx > ix:  return Action(action_type="move", direction="up")
            elif ry < iy:  return Action(action_type="move", direction="right")
            else:          return Action(action_type="move", direction="left")
        return Action(action_type="move", direction="down")

    else:
        if rx == gx and ry == gy: return Action(action_type="drop")
        elif rx < gx:  return Action(action_type="move", direction="down")
        elif rx > gx:  return Action(action_type="move", direction="up")
        elif ry < gy:  return Action(action_type="move", direction="right")
        else:          return Action(action_type="move", direction="left")


# ── LLM call via proxy ────────────────────────────────────────────────────────
def get_llm_action(obs):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a warehouse robot planner. "
                    "Reply with exactly one word: pick, drop, or move."
                ),
            },
            {"role": "user", "content": str(obs.dict())},
        ],
        max_tokens=20,
    )
    content = response.choices[0].message.content.lower()
    if "pick"  in content: return Action(action_type="pick")
    if "drop"  in content: return Action(action_type="drop")
    return Action(action_type="move", direction="right")


# ── Task runner ───────────────────────────────────────────────────────────────
def run_task(task_name: str):
    # [START] — required structured output
    print(f"[START] task={task_name}", flush=True)

    if WarehouseEnv is None or Action is None:
        print(f"[STEP] step=1 reward=0.0", flush=True)
        print(f"[END] task={task_name} score=0.0 steps=1", flush=True)
        return

    try:
        env = WarehouseEnv(task=task_name)
    except TypeError:
        env = WarehouseEnv()

    obs          = env.reset()
    done         = False
    total_reward = 0.0
    step_count   = 0

    while not done and step_count < MAX_STEPS:
        # Try LLM via proxy first; fall back to heuristic only on error
        try:
            action = get_llm_action(obs)
        except Exception as e:
            print(f"# LLM error step {step_count + 1}: {e}", flush=True)
            action = get_heuristic_action(obs, env)

        if action is None:
            action = Action(action_type="move", direction="right")

        obs, reward, done, _ = env.step(action)
        reward_val    = float(reward.value) if hasattr(reward, "value") else float(reward)
        total_reward += reward_val
        step_count   += 1

        # [STEP] — required structured output
        print(f"[STEP] step={step_count} reward={reward_val:.4f}", flush=True)

    score = min(max(total_reward / max(step_count, 1), 0.0), 1.0)

    # [END] — required structured output
    print(f"[END] task={task_name} score={score:.4f} steps={step_count}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
    sys.stdout.flush()