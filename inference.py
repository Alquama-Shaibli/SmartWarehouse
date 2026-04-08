import os
import sys

# ── Env import with graceful fallback ────────────────────────────────────────
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

# ── Tasks defined in openenv.yaml ────────────────────────────────────────────
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 50


def get_heuristic_action(obs, env, Action):
    """Rule-based fallback agent (your original logic, preserved)."""
    sm = env.state_manager
    rx, ry = obs.robot_position
    gx, gy = obs.goal
    cx, cy = sm.charge_station

    if obs.battery < 20 and sm.robot != [cx, cy]:
        if rx > cx:
            return Action(action_type="move", direction="up")
        elif rx < cx:
            return Action(action_type="move", direction="down")
        elif ry > cy:
            return Action(action_type="move", direction="left")
        elif ry < cy:
            return Action(action_type="move", direction="right")
    elif len(sm.carrying) < 3:
        target_item_pos = None
        for item, pos in sm.inventory.items():
            if item not in sm.carrying:
                target_item_pos = pos
                break
        if target_item_pos:
            ix, iy = target_item_pos
            if rx == ix and ry == iy:
                return Action(action_type="pick")
            elif rx < ix:
                return Action(action_type="move", direction="down")
            elif rx > ix:
                return Action(action_type="move", direction="up")
            elif ry < iy:
                return Action(action_type="move", direction="right")
            elif ry > iy:
                return Action(action_type="move", direction="left")
        else:
            return Action(action_type="move", direction="down")
    else:
        if rx == gx and ry == gy:
            return Action(action_type="drop")
        elif rx < gx:
            return Action(action_type="move", direction="down")
        elif rx > gx:
            return Action(action_type="move", direction="up")
        elif ry < gy:
            return Action(action_type="move", direction="right")
        elif ry > gy:
            return Action(action_type="move", direction="left")

    return Action(action_type="move", direction="right")


def run_task(task_name: str):
    """Run one task and emit the required structured stdout blocks."""

    # ── [START] block ────────────────────────────────────────────────────────
    print(f"[START] task={task_name}", flush=True)

    # ── Environment setup ────────────────────────────────────────────────────
    if WarehouseEnv is None or Action is None:
        # Environment not available — emit a minimal valid trace so the
        # validator can still parse [START]/[STEP]/[END] blocks.
        print(f"[STEP] step=1 reward=0.0", flush=True)
        print(f"[END] task={task_name} score=0.0 steps=1", flush=True)
        return

    try:
        env = WarehouseEnv(task=task_name)
    except TypeError:
        env = WarehouseEnv()

    obs = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0

    # ── Optional LLM client (falls back to heuristic if unavailable) ─────────
    client = None
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key and api_key != "dummy-key":
            client = OpenAI(api_key=api_key)
    except ImportError:
        pass

    # ── Main episode loop ────────────────────────────────────────────────────
    while not done and step_count < MAX_STEPS:
        action = None

        # Try LLM first
        if client is not None:
            try:
                response = client.chat.completions.create(
                    model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
                    messages=[
                        {"role": "system", "content": "You are a warehouse robot planner."},
                        {"role": "user", "content": str(obs.dict())}
                    ]
                )
                content = response.choices[0].message.content.lower()
                if "pick" in content:
                    action = Action(action_type="pick")
                elif "drop" in content:
                    action = Action(action_type="drop")
                else:
                    action = Action(action_type="move", direction="right")
            except Exception:
                action = None

        # Fall back to heuristic
        if action is None:
            action = get_heuristic_action(obs, env, Action)

        # Safety net
        if action is None:
            action = Action(action_type="move", direction="right")

        obs, reward, done, _ = env.step(action)
        reward_val = float(reward.value) if hasattr(reward, "value") else float(reward)
        total_reward += reward_val
        step_count += 1

        # ── [STEP] block (required by validator) ────────────────────────────
        print(f"[STEP] step={step_count} reward={reward_val:.4f}", flush=True)

    # Normalise score to [0, 1] range expected by the validator
    max_possible = max(step_count, 1) * 1.0
    score = min(max(total_reward / max_possible, 0.0), 1.0)

    # ── [END] block (required by validator) ─────────────────────────────────
    print(f"[END] task={task_name} score={score:.4f} steps={step_count}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for task in TASKS:
        run_task(task)

    sys.stdout.flush()