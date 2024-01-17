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

from minigrid.minigrid_env import MiniGridEnv, is_slippery

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

    - `MiniGrid-VLavaSlipperyS12-v0`
    - `MiniGrid-VLavaSlipperyS12-v0`
    - `MiniGrid-VLavaSlipperyS12-v0`
    - `MiniGrid-VLavaSlipperyS12-v0`

    """
    def __init__(self,
                randomize_start=True, size=12,
                width=None,
                height=None,
                probability_intended=8/9,
                probability_turn_intended=8/9,
                obstacle_type=Lava,
                goal_reward=1,
                failure_penalty=-1,
                per_step_penalty=0,
                bfs_rewards=False,
                     **kwargs):

        self.obstacle_type = obstacle_type
        self.size = size
        self.probability_intended = probability_intended
        self.probability_turn_intended = probability_turn_intended

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
            max_steps=200,
            # Set this to True for maximum speed
            see_through_walls=False,
            **kwargs
        )

        self.randomize_start = randomize_start
        self.goal_reward = goal_reward
        self.failure_penalty = failure_penalty
        self.bfs_rewards = bfs_rewards
        self.per_step_penalty = per_step_penalty


    def _create_slippery_north(self):
        return SlipperyNorth(probability_intended=self.probability_intended,
                             probability_turn_intended=self.probability_turn_intended)


    def _create_slippery_south(self):
        return SlipperySouth(probability_intended=self.probability_intended,
                             probability_turn_intended=self.probability_turn_intended)


    def _create_slippery_east(self):
        return SlipperyEast(probability_intended=self.probability_intended,
                             probability_turn_intended=self.probability_turn_intended)


    def _create_slippery_west(self):
        return SlipperyWest(probability_intended=self.probability_intended,
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

    def disable_random_start(self):
        self.randomize_start = False

    def place_agent(self, spawn_on_slippery=False, agent_pos=None, agent_dir=0):
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
                if cell is None or (cell.can_overlap() and not isinstance(cell, Lava) and not isinstance(cell, Goal) and (spawn_on_slippery or not is_slippery(cell))):
                    self.agent_pos = np.array((x, y))
                    self.agent_dir = np.random.randint(0, 4)
                    break
        elif agent_dir is None:
            self.agent_pos = np.array((1, 1))
            self.agent_dir = 0
        else:
            self.agent_pos = agent_pos
            self.agent_dir = agent_dir

    def place_goal(self, goal_pos):
        self.goal_pos = goal_pos
        self.put_obj(Goal(), *self.goal_pos)

    def run_bfs(self):
        if self.bfs_rewards:
            self.bfs_reward = self.run_BFS_reward()
            self.bfs_reward = [rew * 0.1 for rew in self.bfs_reward]

    def printGrid(self, init=False):
        grid = super().printGrid(init)

        properties_str = ""

        properties_str += F"ProbTurnIntended:{self.probability_turn_intended}\n"
        properties_str += F"ProbForwardIntended:{self.probability_intended}\n"

        return grid + properties_str

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward - self.per_step_penalty, terminated, truncated, info

class LavaSlipperyPool(LavaSlipperyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        #TODO
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


        self.put_obj(self._create_slippery_south(), w_mid - 3, h_mid - 3)
        self.put_obj(self._create_slippery_south(), w_mid - 2, h_mid - 3)
        self.put_obj(self._create_slippery_south(), w_mid - 1, h_mid - 3)
        self.put_obj(self._create_slippery_south(), w_mid, h_mid - 3)
        self.put_obj(self._create_slippery_south(), w_mid + 1, h_mid - 3)
        self.put_obj(self._create_slippery_south(), w_mid + 2, h_mid - 3)

        self.put_obj(self._create_slippery_east(), w_mid - 3, h_mid - 2)
        self.put_obj(self._create_slippery_east(), w_mid - 3, h_mid - 1)
        self.put_obj(self._create_slippery_east(), w_mid - 3, h_mid)
        self.put_obj(self._create_slippery_east(), w_mid - 3, h_mid + 1)

        self.put_obj(self._create_slippery_west(), w_mid + 2, h_mid - 2)
        self.put_obj(self._create_slippery_west(), w_mid + 2, h_mid - 1)
        self.put_obj(self._create_slippery_west(), w_mid + 2, h_mid)
        self.put_obj(self._create_slippery_west(), w_mid + 2, h_mid + 1)

        self.put_obj(self._create_slippery_north(), w_mid - 3, h_mid + 2)
        self.put_obj(self._create_slippery_north(), w_mid - 2, h_mid + 2)
        self.put_obj(self._create_slippery_north(), w_mid - 1, h_mid + 2)
        self.put_obj(self._create_slippery_north(), w_mid, h_mid + 2)
        self.put_obj(self._create_slippery_north(), w_mid + 1, h_mid + 2)
        self.put_obj(self._create_slippery_north(), w_mid + 2, h_mid + 2)

        self.place_agent(agent_pos=np.array((1, 1)), agent_dir=0)
        self.place_goal(np.array((width - 2, height - 2)))
        self.run_bfs()

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

        self.put_obj(self._create_slippery_east(), w_mid - 2, h_mid - 1)
        self.put_obj(self._create_slippery_east(), w_mid - 2, h_mid)

        self.put_obj(self._create_slippery_west(), w_mid + 1, h_mid - 1)
        self.put_obj(self._create_slippery_west(), w_mid + 1, h_mid)


        self.put_obj(Lava(), w_mid - 1, 1)
        self.put_obj(Lava(), w_mid, 1)

        self.put_obj(self._create_slippery_north(), w_mid - 1, 2)
        self.put_obj(self._create_slippery_north(), w_mid, 2)
        self.put_obj(self._create_slippery_east(), w_mid - 2, 1)
        self.put_obj(self._create_slippery_west(), w_mid + 1, 1)

        self.put_obj(Lava(), w_mid - 1, height - 2)
        self.put_obj(Lava(), w_mid, height - 2)

        self.put_obj(self._create_slippery_south(), w_mid - 1, height - 3)
        self.put_obj(self._create_slippery_south(), w_mid, height - 3)
        self.put_obj(self._create_slippery_east(), w_mid - 2, height - 2)
        self.put_obj(self._create_slippery_west(), w_mid + 1, height - 2)

        self.place_agent(agent_pos=np.array((1, 1)), agent_dir=0)
        self.place_goal(np.array((width - 2, height - 2)))
        self.run_bfs()

class LavaSlipperyCliff(LavaSlipperyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        for i in range(1,5):
            self.grid.horz_wall(3, i, width - 6, Lava)
        for i in range(5,height - 3):
            self.grid.horz_wall(3, i, width - 6, SlipperyNorth)


        self.place_agent(agent_pos=np.array((1, 1)), agent_dir=0)
        self.place_goal(np.array((width - 2, 1)))
        self.run_bfs()

class LavaSlipperyHill(LavaSlipperyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        for i in range(1,height - 1):
            self.grid.horz_wall(1, i, width - 2, SlipperyNorth)
        for i in range(1,5):
            self.grid.horz_wall(3, i, width - 6, Lava)


        self.place_agent(agent_pos=np.array((1, 1)), agent_dir=0, spawn_on_slippery=True)
        self.place_goal(np.array((width - 2, 1)))
        self.run_bfs()

class LavaSlipperyMaze(LavaSlipperyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _gen_grid(self, width, height):
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

        self.place_agent(agent_pos=np.array((width - 2, height - 2)), agent_dir=0)
        self.place_goal(np.array((width - 2, height - 2)))
        self.run_bfs()
