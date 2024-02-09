from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, SlipperyNorth, SlipperyEast, SlipperySouth, SlipperyWest, Ball
from minigrid.envs.adversaries_base import AdversaryEnv
from minigrid.core.tasks import GoTo, DoNothing, PickUpObject, PlaceObject

import numpy as np

class AdversaryDebug(AdversaryEnv):

    """
    ## Description


    ## Registered Configurations

    - `MiniGrid-Adv-8x8-v0`
    - `MiniGrid-AdvLava-8x8-v0`
    - `MiniGrid-AdvSlipperyLava-8x8-v0`
    - `MiniGrid-AdvDebug-8x8-v0`

    """

    def __init__(self, width=7, height=6, generate_wall=True, generate_lava=False, generate_slippery=False ,max_steps: int | None = None, **kwargs):
        if max_steps is None:
            max_steps = 10 * (width * height)**2

        self.generate_wall = generate_wall
        self.generate_lava = generate_lava
        self.generate_slippery = generate_slippery
        super().__init__(
            width=width, height=height, max_steps=max_steps, **kwargs
        )


    def __generate_slippery(self, width, height):
        self.put_obj(Lava(), 2, height - 2)
        self.put_obj(Lava(), width - 2, height - 4)

        self.put_obj(SlipperyEast(), 3, height-2)
        self.put_obj(SlipperyWest(), 1, height-2)
        self.put_obj(SlipperyNorth(), 2, height-3)

        self.put_obj(SlipperyNorth(), width - 2, height-5)
        self.put_obj(SlipperyWest(), width - 3, height-4)
        self.put_obj(SlipperySouth(), width - 2, height-3)


    def __generate_lava(self, width, height):
        self.gap_pos = np.array(
            (
                width // 2,
                height // 2,
            )
        )
        self.grid.vert_wall(self.gap_pos[0], 1, height - 2, Lava)

        # Put a hole in the wall
        self.grid.set(*self.gap_pos, None)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(), width - 2, height - 2)

        blue_ball_pos = (2,3)
        self.put_obj(Ball(), *blue_ball_pos)
        blue_ball = self.grid.get(*blue_ball_pos)

        self.adversaries = {}
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        if self.generate_wall:
            wall_length = 3
            self.grid.horz_wall(width - wall_length - 2, 2, wall_length)
            self.put_obj(SlipperyEast(), width - 3, 1)
            self.put_obj(SlipperyNorth(), 3, height-2)
        elif self.generate_lava:
            self.__generate_lava(width, height)

        elif self.generate_slippery:
            self.__generate_slippery(width, height)


        blue_adv = self.add_adversary(width - 2, 1, "blue", direction=2, tasks=[GoTo((1, 3)),
                                                                                PickUpObject(blue_ball_pos, blue_ball),
                                                                                DoNothing(duration=4),
                                                                                GoTo((width - 2, 1)),
                                                                                PlaceObject((width - 2, 2), blue_ball),
                                                                                DoNothing(duration=2),
                                                                                PickUpObject((width - 2, 2), blue_ball),
                                                                                GoTo((1,3)),
                                                                                PlaceObject(blue_ball_pos, blue_ball)], repeating=True)
        blue_adv.append_task(DoNothing(duration=7))
        blue_adv.insert_task(GoTo((1,2)), 3)



    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
