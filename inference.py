import os
from openai import OpenAI
from warehouse_env.env_core import WarehouseEnv
from warehouse_env.models import Action

# Will use fake API logic if API key isn't provided/valid, or actual LLM
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key"))

env = WarehouseEnv()
obs = env.reset()
done = False
total_reward = 0

print(f"--- Starting inference episode ---")
print(f"Initial State: Robot={obs.robot_position}, Battery={obs.battery}")

while not done and env.state_manager.steps < 50:
    try:
        # LLM based inference logic
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

    except Exception as e:
        # Fallback heuristic if LLM fails
        sm = env.state_manager
        rx, ry = obs.robot_position
        gx, gy = obs.goal
        cx, cy = sm.charge_station

        # Simple pathfinding with states
        if obs.battery < 20 and sm.robot != [cx, cy]:
            if rx > cx:
                action = Action(action_type="move", direction="up")
            elif rx < cx:
                action = Action(action_type="move", direction="down")
            elif ry > cy:
                action = Action(action_type="move", direction="left")
            elif ry < cy:
                action = Action(action_type="move", direction="right")
        elif len(sm.carrying) < 3:
            # Look for an unpicked item
            target_item_pos = None
            for item, pos in sm.inventory.items():
                if item not in sm.carrying:
                    target_item_pos = pos
                    break

            if target_item_pos:
                ix, iy = target_item_pos
                if rx == ix and ry == iy:
                    action = Action(action_type="pick")
                elif rx < ix:
                    action = Action(action_type="move", direction="down")
                elif rx > ix:
                    action = Action(action_type="move", direction="up")
                elif ry < iy:
                    action = Action(action_type="move", direction="right")
                elif ry > iy:
                    action = Action(action_type="move", direction="left")
            else:
                action = Action(action_type="move", direction="down")  # Wandering
        else:
            if rx == gx and ry == gy:
                action = Action(action_type="drop")
            elif rx < gx:
                action = Action(action_type="move", direction="down")
            elif rx > gx:
                action = Action(action_type="move", direction="up")
            elif ry < gy:
                action = Action(action_type="move", direction="right")
            elif ry > gy:
                action = Action(action_type="move", direction="left")

    obs, reward, done, _ = env.step(action)
    total_reward += reward.value

    # Logging
    print(f"Step {env.state_manager.steps} | Action: {action.action_type} {action.direction or ''} | Reward: {reward.value} | Battery: {obs.battery} | Carrying: {sm.carrying}")

print(f"--- Inference Complete ---")
print(f"Final Score: {total_reward}")
print(f"Total Collisions: {env.state_manager.collisions}")
