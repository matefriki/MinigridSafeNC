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
    def __init__(self, size=12, width=None, height=None, probability_forward=3/9, probability_direct_neighbour=2/9, probability_next_neighbour=1/9,    obstacle_type=Lava, version=0 , **kwargs):
        self.obstacle_type = obstacle_type
        self.size = size
        self.version = version
        self.probability_forward = probability_forward
        self.probability_direct_neighbour = probability_direct_neighbour
        self.probability_next_neighbour = probability_next_neighbour

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
            **kwargs
        )
        
    def _create_slippery_north(self):
        return SlipperyNorth(probability_forward=self.probability_forward, 
                                    probability_next_neighbour=self.probability_next_neighbour,
                                    probability_direct_neighbour=self.probability_direct_neighbour)

    
    def _create_slippery_south(self):
        return SlipperySouth(probability_forward=self.probability_forward, 
                                    probability_next_neighbour=self.probability_next_neighbour,
                                    probability_direct_neighbour=self.probability_direct_neighbour)


    def _create_slippery_east(self):
        return SlipperyEast(probability_forward=self.probability_forward, 
                                    probability_next_neighbour=self.probability_next_neighbour,
                                    probability_direct_neighbour=self.probability_direct_neighbour)

    
    def _create_slippery_west(self):
        return SlipperyWest(probability_forward=self.probability_forward, 
                                    probability_next_neighbour=self.probability_next_neighbour,
                                    probability_direct_neighbour=self.probability_direct_neighbour)

    def _place_slippery_lava(self, x, y):
        self.put_obj(Lava(), x, y)
        self.put_obj(self._create_slippery_north(), x, y - 1)
        self.put_obj(self._create_slippery_south(), x, y + 1)
        self.put_obj(self._create_slippery_east(), x + 1, y)
        self.put_obj(self._create_slippery_west(), x - 1, y)

        
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
        self.put_obj(self._create_slippery_north(), w_mid - 3, h_mid - 3)    
        self.put_obj(self._create_slippery_north(), w_mid - 2, h_mid - 3)
        self.put_obj(self._create_slippery_north(), w_mid - 1, h_mid - 3)
        self.put_obj(self._create_slippery_north(), w_mid, h_mid - 3)
        self.put_obj(self._create_slippery_north(), w_mid + 1, h_mid - 3)
        self.put_obj(self._create_slippery_north(), w_mid + 2, h_mid - 3)
        
        self.put_obj(self._create_slippery_west(), w_mid - 3, h_mid - 2)
        self.put_obj(self._create_slippery_west(), w_mid - 3, h_mid - 1)
        self.put_obj(self._create_slippery_west(), w_mid - 3, h_mid)
        self.put_obj(self._create_slippery_west(), w_mid - 3, h_mid + 1)
        
        self.put_obj(self._create_slippery_east(), w_mid + 2, h_mid - 2)
        self.put_obj(self._create_slippery_east(), w_mid + 2, h_mid - 1)
        self.put_obj(self._create_slippery_east(), w_mid + 2, h_mid)
        self.put_obj(self._create_slippery_east(), w_mid + 2, h_mid + 1)
        
        self.put_obj(self._create_slippery_south(), w_mid - 3, h_mid + 2)    
        self.put_obj(self._create_slippery_south(), w_mid - 2, h_mid + 2)
        self.put_obj(self._create_slippery_south(), w_mid - 1, h_mid + 2)
        self.put_obj(self._create_slippery_south(), w_mid, h_mid + 2)
        self.put_obj(self._create_slippery_south(), w_mid + 1, h_mid + 2)
        self.put_obj(self._create_slippery_south(), w_mid + 2, h_mid + 2)
        
        
        
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
        
        self.put_obj(self._create_slippery_west(), w_mid - 2, h_mid - 1)
        self.put_obj(self._create_slippery_west(), w_mid - 2, h_mid)
        
        self.put_obj(self._create_slippery_east(), w_mid + 1, h_mid - 1)
        self.put_obj(self._create_slippery_east(), w_mid + 1, h_mid)
        
        
        self.put_obj(Lava(), w_mid - 1, 1)
        self.put_obj(Lava(), w_mid, 1)
        
        self.put_obj(self._create_slippery_south(), w_mid - 1, 2)
        self.put_obj(self._create_slippery_south(), w_mid, 2)
        self.put_obj(self._create_slippery_west(), w_mid - 2, 1)
        self.put_obj(self._create_slippery_east(), w_mid + 1, 1)
        
        self.put_obj(Lava(), w_mid - 1, height - 2)
        self.put_obj(Lava(), w_mid, height - 2)
        
        self.put_obj(self._create_slippery_north(), w_mid - 1, height - 3)
        self.put_obj(self._create_slippery_north(), w_mid, height - 3)
        self.put_obj(self._create_slippery_west(), w_mid - 2, height - 2)
        self.put_obj(self._create_slippery_east(), w_mid + 1, height - 2)
        
        # Place the agent
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        # Place a goal square 
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

  
    def _env_three(self, width, height):
        w_mid = width // 2
        h_mid = height // 2
    
        self.put_obj(Lava(), w_mid - 2, 1)
        self.put_obj(Lava(), w_mid - 1, 1)
        self.put_obj(Lava(), w_mid, 1)  
        self.put_obj(Lava(), w_mid + 1, 1)  
        self.put_obj(Lava(), w_mid - 2, 2)
        self.put_obj(Lava(), w_mid - 1, 2)
        self.put_obj(Lava(), w_mid, 2) 
        self.put_obj(Lava(), w_mid + 1, 2) 
        self.put_obj(Lava(), w_mid - 2, 2)
        self.put_obj(Lava(), w_mid - 1, 2)
        self.put_obj(Lava(), w_mid, 2) 
        self.put_obj(Lava(), w_mid + 1, 3) 
        self.put_obj(Lava(), w_mid - 2, 3)
        self.put_obj(Lava(), w_mid - 1, 3)
        self.put_obj(Lava(), w_mid, 3) 
        self.put_obj(Lava(), w_mid + 1, 3) 
        
        
        
        self.put_obj(self._create_slippery_south(), w_mid - 2, 4)
        self.put_obj(self._create_slippery_south(), w_mid - 1, 4)
        self.put_obj(self._create_slippery_south(), w_mid, 4)
        self.put_obj(self._create_slippery_south(), w_mid + 1, 4)
        
        self.put_obj(self._create_slippery_south(), w_mid - 2, 5)
        self.put_obj(self._create_slippery_south(), w_mid - 1, 5)
        self.put_obj(self._create_slippery_south(), w_mid, 5)
        self.put_obj(self._create_slippery_south(), w_mid + 1, 5)
        
        self.put_obj(self._create_slippery_south(), w_mid - 2, 6)
        self.put_obj(self._create_slippery_south(), w_mid - 1, 6)
        self.put_obj(self._create_slippery_south(), w_mid, 6)
        self.put_obj(self._create_slippery_south(), w_mid + 1, 6)
        
        self.put_obj(self._create_slippery_south(), w_mid - 2, 7)
        self.put_obj(self._create_slippery_south(), w_mid - 1, 7)
        self.put_obj(self._create_slippery_south(), w_mid, 7)
        self.put_obj(self._create_slippery_south(), w_mid + 1, 7)
        
        
        
        self.randomize_start = False 
        # Place the agent
        if self.randomize_start == True:
            while True:
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)

                cell = self.grid.get(*(x,y))
                if cell is None or (cell.can_overlap() and not isinstance(cell, Lava)):
                    self.agent_pos = np.array((x, y))
                    self.agent_dir = np.random.randint(0, 4)
                    break
        else:
            self.agent_pos = np.array((1, 1))
            self.agent_dir = 0
        print(F"Agent position {self.agent_pos}")
        # Place a goal square 
        self.goal_pos = np.array((width - 2,1))
        self.put_obj(Goal(), *self.goal_pos)

    def create_slippery_lava_line(self, y, x_start, x_end):
        self.put_obj(self._create_slippery_west(), x_start - 1, y)
        self.put_obj(self._create_slippery_east(), x_end + 1 , y)
        self.put_obj(self._create_slippery_north(), x_start - 1, y - 1)
        self.put_obj(self._create_slippery_south(), x_end + 1 , y + 1)
        self.put_obj(self._create_slippery_north(), x_end + 1, y - 1)
        self.put_obj(self._create_slippery_south(), x_start - 1 , y + 1)

        for x in range(x_start, x_end + 1):
            self.put_obj(self._create_slippery_north(), x, y - 1)
            self.put_obj(self._create_slippery_south(), x, y + 1)
            self.put_obj(Lava(), x, y)

    def create_lava_line(self, y, x_start, x_end):
        for x in range(x_start, x_end + 1):
            self.put_obj(Lava(), x, y)


    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """
        # return 1 - 0.9 * (self.step_count / self.max_steps)
        return 100
        # return 100


    def _env_four(self, width, height):
        
        self.create_lava_line(height - 4, 1, 15)        
        self.create_lava_line(height - 4, 20, 28)        
        self.create_slippery_lava_line(height // 2 + 3, 4, 6)        
        self.create_slippery_lava_line(height // 2 + 3, 9, 15)   
        self.create_slippery_lava_line(height // 2 + 3, 18, 24)   

        self.create_lava_line(height // 2 - 1, 1, 9)  
        self.create_lava_line(height // 2 - 1, 12, 28)  
        
        self.create_slippery_lava_line(5, 2, 6)        
        self.create_slippery_lava_line(5, 9, 15)   
        self.create_slippery_lava_line(5, 18, 24)   

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
