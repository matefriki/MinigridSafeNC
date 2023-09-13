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

class LavaSlipperyEnv(MiniGridEnv):
    
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

    - `MiniGrid-VLavaSlipperyS12-v0`
    - `MiniGrid-VLavaSlipperyS12-v0`
    - `MiniGrid-VLavaSlipperyS12-v0`
    - `MiniGrid-VLavaSlipperyS12-v0`

    """
    def __init__(self, size, obstacle_type=Lava, version=0 , **kwargs):
        self.obstacle_type = obstacle_type
        self.size = size
        self.version = version

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
            width=size,
            height=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=False,
            **kwargs
        )
        
    def _place_slippery_lava(self, x, y):
        self.put_obj(Lava(), x, y)
        self.put_obj(SlipperyNorth(), x, y - 1)
        self.put_obj(SlipperySouth(), x, y + 1)
        self.put_obj(SlipperyEast(), x + 1, y)
        self.put_obj(SlipperyWest(), x - 1, y)

        
    def _env_one(self, width, height):
        w_mid = width // 2
        h_mid = height // 2
        
        self.put_obj(Lava(), w_mid - 1, h_mid - 2)
        self.put_obj(Lava(), w_mid - 2, h_mid - 2)
        self.put_obj(Lava(), w_mid, h_mid - 2)
        self.put_obj(Lava(), w_mid + 1, h_mid - 2)
        
        self.put_obj(Lava(), w_mid - 1, h_mid - 1)
        self.put_obj(Lava(), w_mid - 2, h_mid - 1)
        self.put_obj(Lava(), w_mid, h_mid - 1)
        self.put_obj(Lava(), w_mid + 1, h_mid - 1)
       
        self.put_obj(Lava(), w_mid - 1, h_mid)
        self.put_obj(Lava(), w_mid - 2, h_mid)
        self.put_obj(Lava(), w_mid, h_mid)
        self.put_obj(Lava(), w_mid + 1, h_mid)
      
        self.put_obj(Lava(), w_mid - 1, h_mid + 1)
        self.put_obj(Lava(), w_mid - 2, h_mid + 1)
        self.put_obj(Lava(), w_mid, h_mid + 1)
        self.put_obj(Lava(), w_mid + 1, h_mid + 1)
      
       
        # self.grid.vert_wall(split, 1, height - 2, Lava)
        self.put_obj(SlipperyNorth(), w_mid - 3, h_mid - 3)    
        self.put_obj(SlipperyNorth(), w_mid - 2, h_mid - 3)
        self.put_obj(SlipperyNorth(), w_mid - 1, h_mid - 3)
        self.put_obj(SlipperyNorth(), w_mid, h_mid - 3)
        self.put_obj(SlipperyNorth(), w_mid + 1, h_mid - 3)
        self.put_obj(SlipperyNorth(), w_mid + 2, h_mid - 3)
        
        self.put_obj(SlipperyWest(), w_mid - 3, h_mid - 2)
        self.put_obj(SlipperyWest(), w_mid - 3, h_mid - 1)
        self.put_obj(SlipperyWest(), w_mid - 3, h_mid)
        self.put_obj(SlipperyWest(), w_mid - 3, h_mid + 1)
        
        self.put_obj(SlipperyEast(), w_mid + 2, h_mid - 2)
        self.put_obj(SlipperyEast(), w_mid + 2, h_mid - 1)
        self.put_obj(SlipperyEast(), w_mid + 2, h_mid)
        self.put_obj(SlipperyEast(), w_mid + 2, h_mid + 1)
        
        self.put_obj(SlipperySouth(), w_mid - 3, h_mid + 2)    
        self.put_obj(SlipperySouth(), w_mid - 2, h_mid + 2)
        self.put_obj(SlipperySouth(), w_mid - 1, h_mid + 2)
        self.put_obj(SlipperySouth(), w_mid, h_mid + 2)
        self.put_obj(SlipperySouth(), w_mid + 1, h_mid + 2)
        self.put_obj(SlipperySouth(), w_mid + 2, h_mid + 2)
        
        
        
        # Place the agent
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        # Place a goal square 
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        
        
    def _env_two(self, width, height):
        w_mid = width // 2
        h_mid = height // 2
        
        self.put_obj(Lava(), w_mid - 1, h_mid - 1)
        self.put_obj(Lava(), w_mid, h_mid - 1)
        self.put_obj(Lava(), w_mid - 1, h_mid)
        self.put_obj(Lava(), w_mid, h_mid)
        
        self.put_obj(SlipperyWest(), w_mid - 2, h_mid - 1)
        self.put_obj(SlipperyWest(), w_mid - 2, h_mid)
        
        self.put_obj(SlipperyEast(), w_mid + 1, h_mid - 1)
        self.put_obj(SlipperyEast(), w_mid + 1, h_mid)
        
        
        self.put_obj(Lava(), w_mid - 1, 1)
        self.put_obj(Lava(), w_mid, 1)
        
        self.put_obj(SlipperySouth(), w_mid - 1, 2)
        self.put_obj(SlipperySouth(), w_mid, 2)
        self.put_obj(SlipperyWest(), w_mid - 2, 1)
        self.put_obj(SlipperyEast(), w_mid + 1, 1)
        
        self.put_obj(Lava(), w_mid - 1, height - 2)
        self.put_obj(Lava(), w_mid, height - 2)
        
        self.put_obj(SlipperyNorth(), w_mid - 1, height - 3)
        self.put_obj(SlipperyNorth(), w_mid, height - 3)
        self.put_obj(SlipperyWest(), w_mid - 2, height - 2)
        self.put_obj(SlipperyEast(), w_mid + 1, height - 2)
        
        # Place the agent
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        # Place a goal square 
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

  
    def _env_three(self, width, height):
        w_mid = width // 2
        h_mid = height // 2
        
        # self.put_obj(Lava(), w_mid - 1, h_mid - 1)
        # self.put_obj(Lava(), w_mid, h_mid - 1)
        # self.put_obj(Lava(), w_mid - 1, h_mid)
        # self.put_obj(Lava(), w_mid, h_mid)
        
        self.put_obj(Lava(), w_mid - 2, 4)
        self.put_obj(Lava(), w_mid - 1, 4)
        self.put_obj(Lava(), w_mid, 4)  
        self.put_obj(Lava(), w_mid + 1, 4)  
        self.put_obj(Lava(), w_mid - 2, 5)
        self.put_obj(Lava(), w_mid - 1, 5)
        self.put_obj(Lava(), w_mid, 5) 
        self.put_obj(Lava(), w_mid + 1, 5) 
        
        
        self.put_obj(SlipperySouth(), w_mid - 2, height - 2)
        self.put_obj(SlipperySouth(), w_mid - 1, height - 2)
        self.put_obj(SlipperySouth(), w_mid, height - 2)
        self.put_obj(SlipperySouth(), w_mid + 1, height - 2)
        
        self.put_obj(SlipperySouth(), w_mid - 2, height - 3)
        self.put_obj(SlipperySouth(), w_mid - 1, height - 3)
        self.put_obj(SlipperySouth(), w_mid, height - 3)
        self.put_obj(SlipperySouth(), w_mid + 1, height - 3)
        
        self.put_obj(SlipperySouth(), w_mid - 2, height - 4)
        self.put_obj(SlipperySouth(), w_mid - 1, height - 4)
        self.put_obj(SlipperySouth(), w_mid, height - 4)
        self.put_obj(SlipperySouth(), w_mid + 1, height - 4)
        
        
        self.put_obj(SlipperySouth(), w_mid - 2, height - 5)
        self.put_obj(SlipperySouth(), w_mid - 1, height - 5)
        self.put_obj(SlipperySouth(), w_mid, height - 5)
        self.put_obj(SlipperySouth(), w_mid + 1, height - 5)
        
        
        self.put_obj(SlipperySouth(), w_mid - 2, height - 6)
        self.put_obj(SlipperySouth(), w_mid - 1, height - 6)
        self.put_obj(SlipperySouth(), w_mid, height - 6)
        self.put_obj(SlipperySouth(), w_mid + 1, height - 6)
        
        
        
        # Place the agent
        self.agent_pos = np.array((1, height - 2))
        self.agent_dir = 0

        # Place a goal square 
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

    def _env_four(self, width, height):
        
        self.put_obj(Lava(), 3, height - 4)
        self.put_obj(Lava(), 4, height - 4)
        self.put_obj(Lava(), 5, height - 4)
        self.put_obj(Lava(), 6, height - 4)
        self.put_obj(Lava(), 7, height - 4)
        self.put_obj(Lava(), 8, height - 4)
        self.put_obj(Lava(), 9, height - 4)
        
        self.put_obj(SlipperyEast(), 10, height - 4)
        self.put_obj(SlipperyWest(), 2, height - 4)
        
        self.put_obj(SlipperyNorth(), 2, height - 5)
        self.put_obj(SlipperyNorth(), 3, height - 5)
        self.put_obj(SlipperyNorth(), 4, height - 5)
        self.put_obj(SlipperyNorth(), 5, height - 5)
        self.put_obj(SlipperyNorth(), 6, height - 5)
        self.put_obj(SlipperyNorth(), 7, height - 5)
        self.put_obj(SlipperyNorth(), 8, height - 5)
        self.put_obj(SlipperyNorth(), 9, height - 5)
        self.put_obj(SlipperyNorth(), 10, height - 5)
        
        self.put_obj(SlipperySouth(), 2, height - 3)    
        self.put_obj(SlipperySouth(), 3, height - 3)
        self.put_obj(SlipperySouth(), 4, height - 3)
        self.put_obj(SlipperySouth(), 5, height - 3)
        self.put_obj(SlipperySouth(), 6, height - 3)
        self.put_obj(SlipperySouth(), 7, height - 3)
        self.put_obj(SlipperySouth(), 8, height - 3)
        self.put_obj(SlipperySouth(), 9, height - 3)
        self.put_obj(SlipperySouth(), 10, height - 3)
        
        self.put_obj(Lava(), 2, 3)
        self.put_obj(Lava(), 3, 3)
        self.put_obj(Lava(), 4, 3)
        self.put_obj(Lava(), 5, 3)
        self.put_obj(Lava(), 6, 3)
        self.put_obj(Lava(), 7, 3)
        self.put_obj(Lava(), 8, 3)
        
        
        self.put_obj(SlipperyEast(), 9, 3)
        self.put_obj(SlipperyWest(), 1, 3)
        
           
        self.put_obj(SlipperyNorth(), 1, 2)
        self.put_obj(SlipperyNorth(), 2, 2)
        self.put_obj(SlipperyNorth(), 3, 2)
        self.put_obj(SlipperyNorth(), 4, 2)
        self.put_obj(SlipperyNorth(), 5, 2)
        self.put_obj(SlipperyNorth(), 6, 2)
        self.put_obj(SlipperyNorth(), 7, 2)
        self.put_obj(SlipperyNorth(), 8, 2)
        self.put_obj(SlipperyNorth(), 9, 2)
            
        self.put_obj(SlipperySouth(), 1, 4)
        self.put_obj(SlipperySouth(), 2, 4)
        self.put_obj(SlipperySouth(), 3, 4)
        self.put_obj(SlipperySouth(), 4, 4)
        self.put_obj(SlipperySouth(), 5, 4)
        self.put_obj(SlipperySouth(), 6, 4)
        self.put_obj(SlipperySouth(), 7, 4)
        self.put_obj(SlipperySouth(), 8, 4)
        self.put_obj(SlipperySouth(), 9, 4)
        
        
        self.agent_pos = np.array((width - 2, height - 2))
        self.agent_dir = 3
        # Place a goal square 
        self.goal_pos = np.array((1, 1))
        self.put_obj(Goal(), *self.goal_pos)

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)
        
        if self.version == 0:
            self._env_one(width, height)
        elif self.version == 1:
            self._env_two(width, height)
        elif self.version == 2:
            self._env_three(width, height)
        else:
            self._env_four(width, height)
       
    
        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )
