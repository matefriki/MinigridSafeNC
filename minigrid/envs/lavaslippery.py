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
    def __init__(self,
                randomize_start=True, size=12,
                width=None,
                height=None,
                probability_intended=3/9,
                probability_displacement=2/9,
                probability_turn_intended=3/9,
                probability_turn_displacement=2/9,
                obstacle_type=Lava,
                     version=0 ,
                     **kwargs):

        self.obstacle_type = obstacle_type
        self.size = size
        self.version = version
        self.probability_intended = probability_intended
        self.probability_displacement = probability_displacement
        self.probability_turn_intended = probability_turn_intended
        self.probability_turn_displacement = probability_turn_displacement

        if width is not None and height is not None:
            self.width = width
            self.height = height
        elif size is not None:
            self.width = size
            self.height = size
        else:
            raise ValueError(f"Please define either width and height or a size for square environments. The set values are width: {width}, height: {height}, size: {size}.")

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

        self.randomize_start = randomize_start


    def _create_slippery_north(self):
        return SlipperyNorth(probability_intended=self.probability_intended,
                             probability_displacement=self.probability_displacement,
                             probability_turn_displacement=self.probability_turn_displacement,
                             probability_turn_intended=self.probability_turn_intended)


    def _create_slippery_south(self):
        return SlipperySouth(probability_intended=self.probability_intended,
                             probability_displacement=self.probability_displacement,
                             probability_turn_displacement=self.probability_turn_displacement,
                             probability_turn_intended=self.probability_turn_intended)


    def _create_slippery_east(self):
        return SlipperyEast(probability_intended=self.probability_intended,
                             probability_displacement=self.probability_displacement,
                             probability_turn_displacement=self.probability_turn_displacement,
                             probability_turn_intended=self.probability_turn_intended)


    def _create_slippery_west(self):
        return SlipperyWest(probability_intended=self.probability_intended,
                             probability_displacement=self.probability_displacement,
                             probability_turn_displacement=self.probability_turn_displacement,
                             probability_turn_intended=self.probability_turn_intended)

    def _place_slippery_lava(self, x, y):
        self.put_obj(Lava(), x, y)
        self.put_obj(self._create_slippery_north(), x, y - 1)
        self.put_obj(self._create_slippery_south(), x, y + 1)
        self.put_obj(self._create_slippery_east(), x + 1, y)
        self.put_obj(self._create_slippery_west(), x - 1, y)


    def create_slippery_lava_line(self, y, x_start, x_end, no_slippery_left=False, no_slippery_right=False):
        if not no_slippery_left:
            self.put_obj(self._create_slippery_west(), x_start - 1, y)

        if not no_slippery_right:
            self.put_obj(self._create_slippery_east(), x_end + 1 , y)

        for x in range(x_start, x_end + 1):
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

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def place_agent(self, agent_pos=None, agent_dir=0):
        max_tries = 10_000
        num_tries = 0
        if self.randomize_start == True:
            while True:
                num_tries += 1
                if num_tries > max_tries:
                    raise RecursionError("rejection sampling failed in place_obj")
                x = np.random.randint(0, self.width)
                y = np.random.randint(0, self.height)

                cell = self.grid.get(*(x,y))
                if cell is None or (cell.can_overlap() and not isinstance(cell, Lava) and not isinstance(cell, Goal)):
                    self.agent_pos = np.array((x, y))
                    self.agent_dir = np.random.randint(0, 4)
                    break
        elif agent_dir is None:
            self.agent_pos = np.array((1, 1))
            self.agent_dir = 0
        else:
            self.agent_pos = agent_pos
            self.agent_dir = agent_dir

class LavaSlipperyPool(LavaSlipperyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
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

class LavaSlipperyEnv1(LavaSlipperyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

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

class LavaSlipperyEnv2(LavaSlipperyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
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


        self.get_start_position()

        # Place a goal square
        self.goal_pos = np.array((width - 2,1))
        self.put_obj(Goal(), *self.goal_pos)

class LavaSlipperyMaze(LavaSlipperyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _gen_grid(self, width, height):
        # TODO make scalable
        super()._gen_grid(width, height)
        self.create_lava_line(height - 5, 1, 7)
        self.create_lava_line(height - 5, 12, 19)
        self.create_slippery_lava_line(height // 2 + 2, 4, 6)
        self.create_slippery_lava_line(height // 2 + 2, 9, 12)
        self.create_slippery_lava_line(height // 2 + 2, 15, 19, False, True)

        self.create_lava_line(height // 2 - 2, 1, 4)
        self.create_lava_line(height // 2 - 2, 7, 19)

        self.create_slippery_lava_line(4, 1, 3, True)
        self.create_slippery_lava_line(4, 6, 9)
        self.create_slippery_lava_line(4, 15, 19, False, True)

        self.get_start_position(np.array((width - 2, height - 2)))

        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)
