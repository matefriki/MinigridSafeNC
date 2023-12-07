from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import (
    Ball, 
    Box,
    Key,
    Slippery,
    SlipperyEast,
    SlipperySouth,
    SlipperyNorth,
    SlipperyWest,
    Lava,
    Goal,
    Point
 )

from minigrid.minigrid_env import MiniGridEnv

import numpy as np

class LavaFaultyEnv(MiniGridEnv):
    
    """
    ### Description

    The agent has to reach the green goal square at the opposite corner of the
    room, and must pass through a narrow gap in a vertical strip of deadly lava.
    Touching the lava terminate the episode with a zero reward. This environment
    is useful for studying safety and safe exploration.

    ### Mission Space

    Depending on the `obstacle_type` parameter:
    - `Lava`: "avoid the lava and get to the green goal square"
    - otherwise: "find the opening and get to the green goal square"

    ### Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ### Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [gym_minigrid/minigrid.py](gym_minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ### Rewards

    A reward of '1' is given for success, and '0' for failure.

    ### Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent falls into lava.
    3. Timeout (see `max_steps`).

    ### Registered Configurations

    S: size of map SxS.
    V: Version

    - `MiniGrid-LavaFaultyS12-v0`
    
    """
    def __init__(self, size=12, width=None, height=None, faulty_probability=30 ,obstacle_type=Lava, version=0 , **kwargs):
        self.obstacle_type = obstacle_type
        self.size = size
        self.version = version

        if width is not None and height is not None:        
            self.width = width
            self.height = height
        else:
            self.width = size
            self.height = size

        if obstacle_type == Lava:
            mission_space = MissionSpace(
                mission_func=lambda: "avoid the lava and get to the green goal square"
            )
        else:
            mission_space = MissionSpace(
                mission_func=lambda: "find the opening and get to the green goal square"
            )

        super().__init__(
            mission_space=mission_space,
            width=self.width,
            height=self.height,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=False,
            faulty_behavior=True,
            faulty_probability=faulty_probability,
            **kwargs
        )
        
    def _place_slippery_lava(self, x, y):
        self.put_obj(Lava(), x, y)
        self.put_obj(SlipperyNorth(), x, y - 1)
        self.put_obj(SlipperyNorth(), x + 1, y - 1)
        self.put_obj(SlipperyNorth(), x - 1, y - 1)
        self.put_obj(SlipperySouth(), x, y + 1)
        self.put_obj(SlipperySouth(), x + 1, y + 1)
        self.put_obj(SlipperySouth(), x - 1, y + 1)
        self.put_obj(SlipperyEast(), x + 1, y)
        self.put_obj(SlipperyWest(), x - 1, y)

    
    def _place_lava(self, x, y):
        self.put_obj(Lava(), x, y)
        self.put_obj(Lava(), x + 1, y)
        self.put_obj(Lava(), x, y + 1)
        self.put_obj(Lava(), x + 1  , y + 1)
        

        
    def _env_one(self, width, height):
        w_mid = width // 2
        h_mid = height // 2
        
        self._place_lava(w_mid + 2, h_mid - 1)
        self._place_lava(w_mid - 1, h_mid + 1)
        self._place_lava(width - 3 ,2)
        self._place_lava(2, height - 3)
        
        # Place the agent
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        # Place a goal square 
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        
    def create_slippery_lava_line(self, y, x_start, x_end):
        self.put_obj(SlipperyWest(), x_start - 1, y)
        self.put_obj(SlipperyEast(), x_end + 1 , y)
        self.put_obj(SlipperyNorth(), x_start - 1, y - 1)
        self.put_obj(SlipperySouth(), x_end + 1 , y + 1)
        self.put_obj(SlipperyNorth(), x_end + 1, y - 1)
        self.put_obj(SlipperySouth(), x_start - 1 , y + 1)

        for x in range(x_start, x_end + 1):
            self.put_obj(SlipperyNorth(), x, y - 1)
            self.put_obj(SlipperySouth(), x, y + 1)
            self.put_obj(Lava(), x, y)

    def create_lava_line(self, y, x_start, x_end):
        for x in range(x_start, x_end + 1):
            self.put_obj(Lava(), x, y)

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        self._env_one(width, height)
       
    
        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )
