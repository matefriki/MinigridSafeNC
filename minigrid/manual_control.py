#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import Env

from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


class ManualControl:
    def __init__(
        self,
        env: Env,
        seed=None,
        random_agent=False
    ) -> None:
        self.env = env
        self.seed = seed
        self.closed = False
        self.random_agent = random_agent

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            if self.random_agent:
                index = np.random.choice(3, 1)[0]
                action = [Actions.left, Actions.right, Actions.forward][index]
                print(action)
                self.step(action)
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.env.close()
                        break
                    if event.type == pygame.KEYDOWN:
                        event.key = pygame.key.name(int(event.key))
                        self.key_handler(event)

    def step(self, action: Actions):
        _, reward, terminated, truncated, _ = self.env.step(action)
        print(f"step={self.env.step_count}, reward={reward:.2f}")

        if terminated:
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print("truncated!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return
        if key == "f12":
            self.take_screenshot()
            return

        key_to_action = {
            "left": Actions.left,
            "right": Actions.right,
            "up": Actions.forward,
            "space": Actions.toggle,
            "pageup": Actions.pickup,
            "pagedown": Actions.drop,
            "tab": Actions.pickup,
            "left shift": Actions.drop,
            "enter": Actions.done,
        }
        if key in key_to_action.keys():
            action = key_to_action[key]
            self.step(action)
        else:
            print(key)

    def take_screenshot(self):
        import datetime
        filename = f"{datetime.datetime.now().isoformat()}.png"
        print(f"Saving a screenshot to '{filename}'")
        window = self.env.window
        screenshot = pygame.Surface(window.get_size())
        screenshot.blit(window, (0,0))
        pygame.image.save(screenshot, filename)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="gym environment to load",
        choices=gym.envs.registry.keys(),
        default="MiniGrid-MultiRoom-N6-v0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--tile-size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent-view",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default="640",
        help="set the resolution for pygame rendering (width and height)",
    )
    parser.add_argument(
        "--random-agent",
        action="store_true",
        help="make the agent move around randomly"
    )


    args = parser.parse_args()

    env: MiniGridEnv = gym.make(
        args.env_id,
        tile_size=args.tile_size,
        render_mode="human",
        agent_pov=args.agent_view,
        agent_view_size=args.agent_view_size,
        screen_size=args.screen_size,
    )

    # TODO: check if this can be removed
    if args.agent_view:
        print("Using agent view")
        env = RGBImgPartialObsWrapper(env, args.tile_size)
        env = ImgObsWrapper(env)

    manual_control = ManualControl(env, seed=args.seed, random_agent=args.random_agent)
    manual_control.start()
