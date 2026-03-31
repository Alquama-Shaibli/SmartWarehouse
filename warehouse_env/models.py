from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict

class Order(BaseModel):
    id: str
    items: List[str]
    priority: int

class Observation(BaseModel):
    robot_position: Tuple[int, int]
    inventory: Dict[str, Tuple[int, int]]
    orders: List[Order]
    carrying: List[str]
    goal: Tuple[int, int]
    obstacles: List[Tuple[int, int]]
    battery: float

class Action(BaseModel):
    action_type: str  # move, pick, drop, charge
    direction: Optional[str] = None
    item_id: Optional[str] = None

class Reward(BaseModel):
    value: float
