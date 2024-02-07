from __future__ import annotations
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import (
    Slippery,
    SlipperyEast,
    SlipperySouth,
    SlipperyNorth,
    SlipperyWest,
    SlipperyNorthEast,
    SlipperyNorthWest,
    #SlipperySouthEast,
    #SlipperySouthWest,
    Lava,
    Goal,
    Wall,
    Point
 )

from minigrid.minigrid_env import MiniGridEnv, is_slippery

import numpy as np

class WindyCityEnv(MiniGridEnv):
    def __init__(self,
                randomize_start=True, size=10,
                width=24,
                height=22,
                probability_intended=8/9,
                probability_turn_intended=8/9,
                obstacle_type=Lava,
                goal_reward=1,
                failure_penalty=-1,
                per_step_penalty=0,
                dense_rewards=False,
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
        self.dense_rewards = dense_rewards
        self.per_step_penalty = per_step_penalty

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


    def printGrid(self, init=False):
        grid = super().printGrid(init)

        properties_str = ""

        properties_str += F"ProbTurnIntended:{self.probability_turn_intended}\n"
        properties_str += F"ProbForwardIntended:{self.probability_intended}\n"

        return grid + properties_str

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward - self.per_step_penalty, terminated, truncated, info

    def _place_building(self, col, row, width, height, obj_type=Wall):
        for i in range(col, width + col):
            self.grid.vert_wall(i, row, height, obj_type=obj_type)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        for i in range(1, height - 1):
            self.grid.horz_wall(1, i, width-2, obj_type=SlipperyNorthEast("white"))

        self._place_building(13, 1, 4, 2)
        self.grid.vert_wall(12, 1, 2, obj_type=SlipperyNorth("white"))
        self.grid.horz_wall(13, 3, 4, obj_type=SlipperyEast("white"))
        self.grid.vert_wall(17, 1, 2, obj_type=SlipperyNorth("white"))

        self._place_building(7,  3, 3, 4)
        self.grid.vert_wall(6, 3, 4, obj_type=SlipperyNorth("white"))
        self.grid.vert_wall(10, 3, 4, obj_type=SlipperyNorth("white"))
        self.grid.horz_wall(7, 2, 3, obj_type=SlipperyEast("white"))
        self.grid.horz_wall(7, 7, 3, obj_type=SlipperyEast("white"))

        self._place_building(15, 7, 6, 4)
        self.grid.vert_wall(14, 7, 4, obj_type=SlipperyNorth("white"))
        self.grid.vert_wall(14, 9, 2, obj_type=Wall)
        self.grid.vert_wall(20, 7, 4, obj_type=SlipperyNorth("white"))
        self.grid.vert_wall(13, 9, 2, obj_type=SlipperyNorth("white"))
        self.grid.horz_wall(15, 6, 5, obj_type=SlipperyEast("white"))
        self.grid.horz_wall(14, 11, 6, obj_type=SlipperyEast("white"))


        self._place_building(5, 11, 5, 6)
        self.grid.vert_wall(4, 11, 6, obj_type=SlipperyNorth("white"))
        self.grid.vert_wall(10, 11, 6, obj_type=SlipperyNorth("white"))
        self.grid.horz_wall(5, 17, 5, obj_type=SlipperyEast("white"))
        self.grid.horz_wall(5, 10, 5, obj_type=SlipperyWest("white"))
        self.grid.horz_wall(6, 9, 4, obj_type=SlipperyWest("white"))
        self.grid.vert_wall(9, 7, 4, obj_type=SlipperySouth("white"))

        self._place_building(21, 13, 2, 5)
        self.grid.vert_wall(20, 13, 5, obj_type=SlipperyNorth("white"))
        self.grid.horz_wall(21, 12, 2, obj_type=SlipperyEast("white"))
        self.grid.horz_wall(21, 18, 2, obj_type=SlipperyEast("white"))




        self.place_agent(agent_pos=np.array((1, height -2)), agent_dir=0, spawn_on_slippery=True)
        self.place_goal(np.array((width - 2, 1)))
        if self.dense_rewards: self.run_bfs()
