import random

class StateManager:
    def __init__(self):
        self.grid_size = 6
        self.reset()

    def reset(self):
        self.robot = [0, 0]
        self.goal = (5, 5)
        self.charge_station = (0, 5)

        self.inventory = {
            "item1": (2, 2),
            "item2": (3, 1),
            "item3": (4, 4),
        }

        self.orders = [
            {"id": "order1", "items": ["item1", "item2"], "priority": 1},
            {"id": "order2", "items": ["item3"], "priority": 2},
        ]

        self.carrying = []
        self.battery = 100
        self.steps = 0
        self.collisions = 0

        self.update_obstacles()
        return self.get_state()

    def update_obstacles(self):
        self.obstacles = [(random.randint(0,5), random.randint(0,5)) for _ in range(2)]

    def move(self, direction):
        x, y = self.robot

        if direction == "up": x -= 1
        elif direction == "down": x += 1
        elif direction == "left": y -= 1
        elif direction == "right": y += 1

        if (x, y) in self.obstacles or not (0 <= x < 6 and 0 <= y < 6):
            self.collisions += 1
            return False

        self.robot = [x, y]
        self.battery -= 1
        return True

    def get_state(self):
        return {
            "robot_position": tuple(self.robot),
            "inventory": self.inventory,
            "orders": self.orders,
            "carrying": self.carrying,
            "goal": self.goal,
            "obstacles": self.obstacles,
            "battery": self.battery,
        }
