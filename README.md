---
title: Smart Warehouse Env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
app_file: server/app.py
pinned: false
---
# 🤖 Smart Warehouse OpenEnv

A **production-grade OpenEnv environment** simulating real-world warehouse automation, where AI agents must optimize **inventory management, order fulfillment, navigation, and energy usage**.

---

## 🚀 Overview

This environment models a **robot operating inside a logistics warehouse**, similar to modern fulfillment centers.

The agent must:

* Navigate a dynamic grid
* Pick and deliver items for multiple orders
* Avoid obstacles
* Manage battery constraints
* Optimize efficiency under real-world conditions

---

## 🌍 Real-World Motivation

Modern warehouses (e.g., Amazon, Flipkart logistics) rely on autonomous robots to:

* Handle multi-order fulfillment
* Navigate crowded environments
* Optimize energy usage
* Minimize delays and collisions

This environment captures these **core operational challenges**, making it a realistic benchmark for AI agents.

---

## 🧠 Core Challenges

The agent must solve a **multi-objective optimization problem**:

* 📦 **Order Fulfillment** → Complete multiple orders with dependencies
* 🧭 **Path Planning** → Navigate efficiently in a grid
* ⚡ **Battery Management** → Recharge strategically
* 🚧 **Obstacle Avoidance** → Handle dynamic hazards
* 🎯 **Task Prioritization** → Respect order priorities

---

## ⚙️ Environment API (OpenEnv Compliant)

### 🔹 `GET /reset`

Resets the environment to the initial state.

### 🔹 `POST /step`

Takes an action and updates the environment.

Example:

```json
{
  "action_type": "move",
  "direction": "right"
}
```

### 🔹 `GET /state`

Returns the full current environment state.

---

## 🧾 Observation Space

```json
{
  "robot_position": [x, y],
  "inventory": { "item": [x, y] },
  "orders": [{ "id": "...", "items": [], "priority": int }],
  "carrying": [],
  "goal": [x, y],
  "obstacles": [[x, y]],
  "battery": float
}
```

---

## 🎮 Action Space

| Action   | Description                     |
| -------- | ------------------------------- |
| `move`   | Move robot (up/down/left/right) |
| `pick`   | Pick item at current location   |
| `drop`   | Deliver items at goal           |
| `charge` | Recharge battery                |

---

## 🏆 Reward Design

A **continuous, normalized reward function (0.0 → 1.0)** ensures meaningful learning:

* `+1.0` → All orders completed
* `+0.5` → Order completion progress
* `+0.3` → Successful item pickup
* `+0.2` → Strategic charging
* `-0.01` → Movement cost (efficiency)
* `-0.3` → Collision penalty
* `-1.0` → Battery depletion

👉 Rewards are **normalized for stable agent learning**

---

## 📊 Info Signals (Explainability)

Each step returns:

```json
{
  "steps": int,
  "battery": float,
  "collisions": int,
  "carrying": [],
  "remaining_orders": int,
  "success": boolean
}
```

This provides **full transparency into agent performance**.

---

## 🧪 Task Levels

### 🟢 Easy

* Single order
* No obstacles
* Full battery
* Focus: basic navigation

---

### 🟡 Medium

* Multi-item order
* Static obstacles
* Moderate battery
* Focus: planning + efficiency

---

### 🔴 Hard

* Multiple orders with priorities
* Dynamic obstacles
* Limited battery
* Focus: full real-world complexity

---

## 📏 Evaluation & Grading

Each task includes a **deterministic grader (0.0 → 1.0)**:

* ✔ Collision avoidance
* ✔ Battery efficiency
* ✔ Order completion ratio

Ensures:

* Reproducible evaluation
* Fair comparison between agents

---

## 🤖 Baseline Agent (Inference)

Includes a **heuristic-based baseline agent** that:

* Moves toward goals
* Manages battery levels
* Executes valid action sequences

Compatible with **OpenAI API for LLM-based agents**.

---

## 🐳 Deployment

### Build & Run

```bash
docker build -t smart-warehouse-env .
docker run -p 7860:7860 smart-warehouse-env
```

---

## 🔗 Live API

Access deployed environment:

```
https://alquamashaibli-smart-warehouse-env.hf.space
```

Swagger Docs:

```
/docs
```

---

## 📦 Project Structure

```
.
├── warehouse_env.py
├── state_manager.py
├── models.py
├── inference.py
├── easy.py
├── medium.py
├── hard.py
├── *_grader.py
├── openenv.yaml
├── app.py
├── Dockerfile
└── README.md
```

---

## 🧩 Why This Stands Out

✔ Real-world simulation (not a toy problem)
✔ Multi-objective optimization
✔ Continuous reward shaping
✔ Deterministic grading
✔ OpenEnv compliant API
✔ Scalable to LLM-based agents

---

## 🏁 Conclusion

This environment provides a **high-fidelity simulation of warehouse automation**, enabling research and benchmarking of AI agents in **complex, real-world decision-making scenarios**.

---