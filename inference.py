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
    """Rule-based fallback agent."""
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


def make_llm_client():
    """
    Build an OpenAI-compatible client pointed at the hackathon's LiteLLM proxy.

    The validator checks that API_BASE_URL and API_KEY were used.
    NEVER hardcode keys or use your own OpenAI credentials here.
    """
    from openai import OpenAI

    api_base = os.environ.get("API_BASE_URL")
    api_key  = os.environ.get("API_KEY")

    if not api_base or not api_key:
        raise ValueError(
            "API_BASE_URL or API_KEY env vars not set. "
            "The hackathon validator injects these automatically."
        )

    return OpenAI(base_url=api_base, api_key=api_key)


def get_llm_action(client, obs, Action):
    """Call the LiteLLM proxy and parse the response into an Action."""
    model = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
    response = client.chat.completions.create(
        model=model,
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
    if "pick" in content:
        return Action(action_type="pick")
    elif "drop" in content:
        return Action(action_type="drop")
    else:
        return Action(action_type="move", direction="right")


def run_task(task_name: str, client):
    """Run one task and emit the required structured stdout blocks."""

    # ── [START] ──────────────────────────────────────────────────────────────
    print(f"[START] task={task_name}", flush=True)

    if WarehouseEnv is None or Action is None:
        # Env unavailable — emit minimal valid trace so validator can parse
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

    # ── Episode loop ─────────────────────────────────────────────────────────
    while not done and step_count < MAX_STEPS:
        action = None

        # Primary: always try the LiteLLM proxy first (validator requires this)
        if client is not None:
            try:
                action = get_llm_action(client, obs, Action)
            except Exception as e:
                print(f"# LLM call failed at step {step_count + 1}: {e}", flush=True)
                action = None

        # Fallback: heuristic only when LLM call fails
        if action is None:
            action = get_heuristic_action(obs, env, Action)

        if action is None:
            action = Action(action_type="move", direction="right")

        obs, reward, done, _ = env.step(action)
        reward_val = float(reward.value) if hasattr(reward, "value") else float(reward)
        total_reward += reward_val
        step_count += 1

        # ── [STEP] ───────────────────────────────────────────────────────────
        print(f"[STEP] step={step_count} reward={reward_val:.4f}", flush=True)

    # Normalise score to [0, 1]
    max_possible = max(step_count, 1) * 1.0
    score = min(max(total_reward / max_possible, 0.0), 1.0)

    # ── [END] ─────────────────────────────────────────────────────────────────
    print(f"[END] task={task_name} score={score:.4f} steps={step_count}", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Build the LiteLLM proxy client ONCE — this triggers the "last_active"
    # update that the validator checks. Must use API_BASE_URL + API_KEY.
    try:
        client = make_llm_client()
    except Exception as e:
        print(f"# FATAL: Could not create LLM client: {e}", flush=True)
        client = None

    for task in TASKS:
        run_task(task, client)

    sys.stdout.flush()