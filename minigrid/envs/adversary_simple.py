from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, SlipperyNorth, SlipperyEast, SlipperySouth, SlipperyWest, Box
from minigrid.minigrid_env import MiniGridEnv

import numpy as np

class AdversaryEnv(MiniGridEnv):

    """
    ## Description


    ## Registered Configurations

    - `MiniGrid-Adv-8x8-v0`

    """

    def __init__(self, width=7, height=6, max_steps: int | None = None, **kwargs):
        if max_steps is None:
            max_steps = 10 * (width * height)**2
        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space, width=width, height=height, max_steps=max_steps, **kwargs
        )

    @staticmethod
    def _gen_mission():
        return "use the key to open the door and then get to the goal"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        self.adversaries = {}
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        wall_length = 3
        self.grid.horz_wall(width - wall_length - 2, 2, wall_length)
        self.put_obj(SlipperyEast(), width - 3, 1)
            
        self.add_adversary(width - 2, 1, "blue", direction=2,)

        self.mission = "use the key to open the door and then get to the goal"

    
    def step(self, action):
        
        obs, reward, terminated, truncated, info = super().step(action)
      
        blocked_positions = [adv.cur_pos for adv in self.adversaries.values()]
        agent_pos = self.agent_pos
        
        for adversary in self.adversaries.values(): 
            adversary_action = self.get_adversary_action(adversary)
            self.move_adversary(adversary, adversary_action, blocked_positions, agent_pos)
            
        return obs, reward, terminated, truncated, info

    def get_adversary_action(self, adversary):
        return adversary.task_manager.get_best_action(adversary.cur_pos, adversary.dir_vec(), adversary.carrying, self)
    
    # Moves the adversary according to current policy, code copy pasted from minigrid.step      
    def move_adversary(self, adversary, action, blocked_positions, agent_pos):
        # fetch current location and forward location
        cur_pos = adversary.cur_pos
        current_cell = self.grid.get(*adversary.cur_pos)
        fwd_pos = cur_pos + adversary.dir_vec()
        fwd_cell = self.grid.get(*fwd_pos)


        if action == self.actions.left:
            adversary.adversary_dir -= 1
            if adversary.adversary_dir < 0:
                adversary.adversary_dir += 4

        # Rotate right
        elif action == self.actions.right:
            adversary.adversary_dir = (adversary.adversary_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward and isinstance(current_cell, SlipperyNorth):
            print("TODO: actual slippery obs, for now: Teleporting the agent")
            # TODO also what if blocked, see below
            adversary.cur_pos = (3, 3)

        elif action == self.actions.forward and not isinstance(current_cell, SlipperyNorth):
            if (fwd_pos == agent_pos).all():
                self.reward -= self.collision_penalty
                self.safety_violations_this_episode += 1
                

            elif (fwd_cell is None or fwd_cell.can_overlap()) and not tuple(fwd_pos) in blocked_positions:
                if isinstance(fwd_cell, Box):
                    self.grid.set_background(*fwd_pos,fwd_cell)
                    self.background_boxes[tuple(fwd_pos)] = fwd_cell  # np.array is not hashable
                adversary.cur_pos = tuple(fwd_pos)

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if adversary.carrying is None:
                    adversary.carrying = fwd_cell
                    adversary.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and adversary.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], adversary.carrying)
                adversary.carrying.cur_pos = fwd_pos
                adversary.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        # finally update the env with these changes

        self.grid.set(*cur_pos, None)
        self.grid.set(*adversary.cur_pos, adversary)