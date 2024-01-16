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
import random

class LavaFaultyEnv(MiniGridEnv):
    """

    ### Registered Configurations

    S: size of map SxS.
    V: Version

    - `MiniGrid-LavaFaultyS12-v0`

    """
    def __init__(self,
                size=12,
                width=None,
                height=None,
                fault_probability=0.1,
                faulty_behavior=True,
                obstacle_type=Lava,
                randomize_start=True,
                **kwargs):

        self.obstacle_type = obstacle_type
        self.size = size
        self.fault_probability = fault_probability
        self.faulty_behavior = faulty_behavior
        self.previous_action = None

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
            max_steps=200,
            # Set this to True for maximum speed
            see_through_walls=False,
            **kwargs
        )

    def fault(self):
        return True if random.random() < self.fault_probability else False

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.step_count > 0 and self.fault():
            action = self.previous_action
        self.previous_action = action
        return super().step(action)

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        self.previous_action = None
        return super().reset(**kwargs)

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5
        # Create an empty grid
        self.grid = Grid(width, height)

        gap = 5
        for row in range(1, height - 1):
            if row < (height - gap):
                self.grid.horz_wall(1, row, width - gap - row, Lava)
        for i, col in enumerate(reversed(range(1, width - 1))):
            self.grid.vert_wall(col, gap + i, None, Lava)

        self.grid.wall_rect(0, 0, width, height)

        self.agent_pos = np.array((1, height - 2))
        self.agent_dir = 3

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )
        self.put_obj(Goal(), width - 2, 1)

    def printGrid(self, init=False):
        grid = super().printGrid(init)

        properties_str = ""

        if self.faulty_behavior:
            properties_str += F"FaultProbability:{self.fault_probability}\n"

        return  grid + properties_str
