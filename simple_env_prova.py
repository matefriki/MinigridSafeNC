from __future__ import annotations

# import sys
# import gymnasium
# sys.modules["gym"] = gymnasium

import numpy as np
import random
from collections import deque
from gymnasium import spaces
import gymnasium
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure

import torch
import torch.nn as nn
import os, subprocess, time

from minigrid.core.constants import COLORS, COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import (
    Ball,
    Door,
    Key,
    Slippery,
    SlipperyEast,
    SlipperySouth,
    SlipperyNorth,
    SlipperyWest,
    Lava,
    Goal,
    Point,
    Wall
 )
from minigrid.minigrid_env import MiniGridEnv, is_slippery
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_circle
from minigrid.core.world_object import WorldObj
from minigrid.core.actions import Actions
from minigrid.wrappers import ImgObsWrapper



class ColoredGoal(WorldObj):
    def __init__(self, color="red"):
        super().__init__("goal", color)

    def can_overlap(self):
        return True  # The agent can overlap with the goal

    def render(self, img):
        # Render the goal with the specified color
        c = COLORS[self.color]
        fill_coords(img, point_in_circle(0.5, 0.5, 0.3), c)


class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical separation wall
        for i in range(0, height):
            self.grid.set(5, i, Wall())
        
        # Place the door and key
        self.grid.set(5, 6, Door(COLOR_NAMES[1], is_locked=True))
        self.grid.set(np.random.randint(1,5), np.random.randint(1,height-1), Key(COLOR_NAMES[1]))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"



class RandomMazeEnv(MiniGridEnv):
    def __init__(
        self,
        size=15,
        wall_fraction=0.33,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        probability_intended=8/9,
        probability_turn_intended=8/9,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.wall_fraction = wall_fraction
        self.probability_intended = probability_intended
        self.probability_turn_intended = probability_turn_intended

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            agent_view_size=size,
            **kwargs,
        )
        self.goal_rewards = {
            "red": 1.0,
            "green": 1.0,
            "blue": 1.0
        }

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # Initialize an empty grid
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)  # Create outer walls

        # Generate a random maze
        maze = self.generate_maze_dfs(width,height)

        # Populate the grid with walls and empty cells
        maze[int(height/2)-1][int(height/2)+1] = 1
        for y in range(height):
            for x in range(width):
                if maze[y][x] == 1:
                    self.grid.set(x, y, Wall())
        n_slippery_tiles = int(width*height/5)
        n_slippery_tiles = 0
        for i in range(n_slippery_tiles):
            pos_x, pos_y = int(np.random.randint(1,width-1)), int(np.random.randint(1,height-1))
            direction_slippery = np.random.randint(0,4)
            if maze[pos_y][pos_x] != 1:
                if direction_slippery == 0:
                    self.grid.horz_wall(pos_x, pos_y, 1, SlipperyNorth(probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
                if direction_slippery == 1:
                    self.grid.horz_wall(pos_x, pos_y, 1, SlipperySouth(probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
                if direction_slippery == 2:
                    self.grid.horz_wall(pos_x, pos_y, 1, SlipperyEast(probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
                if direction_slippery == 3:
                    self.grid.horz_wall(pos_x, pos_y, 1, SlipperyWest(probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))


        # Place the agent in a random accessible position
        self.place_agent(top=(1, 1), size=(width-2, height-2))

        # Place the goal in another random accessible position
        # self.place_obj(Goal(), top=(1, 1), size=(width-2, height-2))
        # self.place_obj(ColoredGoal("red"), top=(1, 1), size=(width-2, height-2))
        red_goal = ColoredGoal("red")
        # self.goal_pos = self.place_obj(red_goal, top=(1, 1), size=(width-2, height-2)) # this line places the goal randomly
        self.goal_pos = self.place_obj(red_goal, top=(1, 2), size=(1, 1)) # this line places the goal at an exact postion
        # self.place_obj(ColoredGoal("green"), top=(1, 1), size=(width-2, height-2))
        # self.place_obj(ColoredGoal("blue"), top=(1, 1), size=(width-2, height-2))
        self.place_obj(Ball("green"), top=(1, 1), size=(width-2, height-2))
        self.place_obj(Ball("blue"), top=(1, 1), size=(width-2, height-2))
        # self.put_obj(Lava(), 5, 5)
        # self.grid.horz_wall(2,5,3,SlipperyNorth)

        self.distance_to_goal_grid = self.generate_distance_to_goal(maze, self.goal_pos)
        

    def step(self, action):
        
        pos_x, pos_y = self.agent_pos[0], self.agent_pos[1]
        # print(f"before: {self.agent_pos=}")
        prev_distance = self.distance_to_goal_grid[pos_y][pos_x]
        
        obs, reward, done, truncated, info = super().step(action)

        pos_x, pos_y = self.agent_pos[0], self.agent_pos[1]
        curr_distance = self.distance_to_goal_grid[pos_y][pos_x]

        # print(f"after: {self.agent_pos=}")
        # print(f"{prev_distance=}, {curr_distance=}")

        # Check if the agent is on a goal
        cell = self.grid.get(*self.agent_pos)
        if cell and isinstance(cell, Goal):
            goal_color = cell.color
            reward += self.goal_rewards.get(goal_color, 0.0)  # Add reward based on goal color
            done = True  # End the episode when a goal is reached

        reward += -0.01 # reward for being late
        reward += (prev_distance - curr_distance)*0.1 # reward for approaching the goal

        # progress_reward, self.last_distance = compute_progress_reward(
        #     self.agent_pos, self.goal_pos, self.last_distance
        # )
        
        # # Combine rewards
        # reward += progress_reward

        return obs, reward, done, truncated, info
    
    def generate_maze_dfs(self, width, height):
        # Initialize a grid filled with walls
        original_width, original_height = width, height
        width, height = int(width/2), int(height/2)

        maze = [[1 for _ in range(width)] for _ in range(height)]
        def carve_passages_from(x, y):
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(directions)  # Randomize direction order
            
            for dx, dy in directions:
                nx, ny = x + 2 * dx, y + 2 * dy  # Move two cells at a time
                if 0 <= nx < width and 0 <= ny < height and maze[nx][ny] == 1:
                    # Carve a path
                    maze[x + dx][y + dy] = 0
                    # maze[x + dx][min(width-1, y + dy+1)] = 0
                    # maze[x + dx][max(0, y + dy-1)] = 0
                    # maze[min(height-1, x + dx+1)][y + dy] = 0
                    # maze[max(0, x + dx-1)][y + dy] = 0
                    maze[nx][ny] = 0
                    carve_passages_from(nx, ny)
        
        # Start carving from a random cell
        start_x, start_y = random.randrange(1, width, 2), random.randrange(1, height, 2)
        maze[start_x][start_y] = 0
        carve_passages_from(start_x, start_y)

        width, height = original_width, original_height
        maze_large = [[1 for _ in range(width)] for _ in range(height)]
        for i in range(width):
            for j in range(height):
                maze_large[i][j] = maze[min(int(width/2)-1, int(i/2))][min(int(height/2)-1, int(j/2))]
        for i in range(1,width):
            maze_large[1][i] = 0
            maze_large[i][1] = 0
            # print((height-1), ((height-1))%4 )
            if ((height-1))%4 == 2:
                maze_large[i][height-2] = 0
                maze_large[width-2][i] = 0
                maze_large[i][height-3] = 0
                maze_large[width-3][i] = 0
        ### comment away from here
        # maze_large = [[0 for _ in range(width)] for _ in range(height)]
        # for i in range(len(maze_large)):
        #     maze_large[i][0] = 1
        #     maze_large[0][i] = 1
        ### until here
        for i in range(len(maze_large)):
            maze_large[i][-1] = 1
            maze_large[-1][i] = 1



        return maze_large

        
        return maze
    
    def generate_distance_to_goal(self, maze, goal):
        goal_x, goal_y = goal
        width, height = len(maze[0]), len(maze)
        for i in range(width):
            for j in range(height):
                if maze[i][j] == 1:
                    maze[i][j] = -2 # wall
                else:
                    maze[i][j] = -1 # not visited
        maze[goal_y][goal_x] = 0

        queue = deque([(goal_y, goal_x)])

        while queue:
            node = queue.popleft()
            node_x, node_y = node
            current_dist = maze[node_x][node_y]
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dx, dy in directions:
                nx, ny = node_x + dx, node_y + dy  # Move one cell at a time
                if 0 <= nx < width and 0 <= ny < height and maze[nx][ny] == -1:
                    maze[nx][ny] = current_dist + 1
                    queue.append((nx,ny))
        return maze
            



                
    
    def printGrid(self, init=False):
        grid = super().printGrid(init)

        properties_str = ""

        properties_str += F"ProbTurnIntended:{self.probability_turn_intended}\n"
        properties_str += F"ProbForwardIntended:{self.probability_intended}\n"

        return grid + properties_str
    
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        return obs




class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))




# Custom callback for saving the model every 20k steps
class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            save_file = os.path.join(self.save_path, f"model_{self.n_calls}_steps")
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls}")
        return True


def train():
    policy_kwargs = dict(
        features_extractor_class=MinigridFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    # Step 1: Set up directories for saving models and logs
    log_dir = "./ppo_minigrid_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Step 2: Create the environment
    env = RandomMazeEnv(render_mode="rgb_array", size=13)
    env = ImgObsWrapper(env)
    env = Monitor(env)  # Monitor the environment to log rewards
    env = DummyVecEnv([lambda: env])  # Vectorize the environment

    # Step 3: Configure TensorBoard logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    # Step 4: Initialize the model
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.set_logger(new_logger)

    # Step 5: Set up the callback for periodic model saving
    save_callback = SaveModelCallback(save_freq=20000, save_path=log_dir)

    # Step 6: Train the model and save checkpoints
    model.learn(total_timesteps=int(5e8), callback=save_callback)

    # Final save
    model.save(os.path.join(log_dir, "ppo_minigrid_final"))
    print("Final model saved successfully.")

    return


# def train():
    

#     policy_kwargs = dict(
#     features_extractor_class=MinigridFeaturesExtractor,
#     features_extractor_kwargs=dict(features_dim=128),
#     )

#     # Step 3: Set up directories for saving models and logs
#     log_dir = "./ppo_minigrid_logs/"
#     os.makedirs(log_dir, exist_ok=True)
    

#     env = RandomMazeEnv(render_mode="rgb_array", size = 13)
#     env = ImgObsWrapper(env)

#     model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
#     model.learn(5e6)

#     model.save(os.path.join(log_dir, "ppo_minigrid"))
#     print("Model saved successfully.")

#     return

#     # Step 6: Train the model
#     try:
#         model.learn(total_timesteps=2000, callback=eval_callback)  # Train for 200k timesteps
#         model.save(os.path.join(log_dir, "ppo_minigrid_longer"))
#         print("Model saved successfully.")
#     except KeyboardInterrupt:
#         print("Training interrupted. Saving the model...")
#         model.save(os.path.join(log_dir, "ppo_minigrid_interrupt"))

#     # Step 7: Evaluate the trained model
#     obs = eval_env.reset()
#     for _ in range(1000):  # Run the trained model for 1000 steps
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, info = eval_env.step(action)
        
#         # # Directly call the render method of the base environment
#         # if hasattr(eval_env, "env"):
#         #     eval_env.env.render()  # Call render without extra arguments
#         # else:
#         #     eval_env.render()  # Fallback if no nested env

#         if done:
#             print("Episode reward:", reward)
#             obs = eval_env.reset()

#     eval_env.close()
    


class BfsModel():
    def __init__(self, env):
        self.env = env
    
    def predict(self, obs, deterministic=True):
        pos_x, pos_y = self.env.unwrapped.agent_pos
        fwd_pos_x, fwd_pos_y = self.env.unwrapped.front_pos
        # print(f"Inside: {self.env.unwrapped.agent_pos=}, {self.env.unwrapped.front_pos=}\n")
        width = len(self.env.unwrapped.distance_to_goal_grid[0])
        height = len(self.env.unwrapped.distance_to_goal_grid)
        if fwd_pos_x < 0 or fwd_pos_x >= width or fwd_pos_y < 0 or fwd_pos_y >= height:
            return 0, 0
        # print(f"{fwd_pos_x=}, {fwd_pos_y=}")
        curr_distance = self.env.unwrapped.distance_to_goal_grid[pos_y][pos_x]
        fwd_distance = self.env.unwrapped.distance_to_goal_grid[fwd_pos_y][fwd_pos_x]
        if fwd_distance >= curr_distance or fwd_distance == -2 :
            return 0, 0
        return 2, 2

        moves = [0, 1, 2] # left, right, forward
        curr_dir = self.env.unwrapped.agent_dir
        print(pos_x, pos_y, curr_dir)
        print(self.env.unwrapped.front_pos)

class BfsModel():
    def __init__(self, env):
        self.env = env
    
    def predict(self, obs, deterministic=True):
        pos_x, pos_y = self.env.unwrapped.agent_pos
        fwd_pos_x, fwd_pos_y = self.env.unwrapped.front_pos
        # print(f"Inside: {self.env.unwrapped.agent_pos=}, {self.env.unwrapped.front_pos=}\n")
        width = len(self.env.unwrapped.distance_to_goal_grid[0])
        height = len(self.env.unwrapped.distance_to_goal_grid)
        if fwd_pos_x < 0 or fwd_pos_x >= width or fwd_pos_y < 0 or fwd_pos_y >= height:
            return 0, 0
        # print(f"{fwd_pos_x=}, {fwd_pos_y=}")
        curr_distance = self.env.unwrapped.distance_to_goal_grid[pos_y][pos_x]
        fwd_distance = self.env.unwrapped.distance_to_goal_grid[fwd_pos_y][fwd_pos_x]
        if fwd_distance >= curr_distance or fwd_distance == -2 :
            return 0, 0
        return 2, 2

        moves = [0, 1, 2] # left, right, forward
        curr_dir = self.env.unwrapped.agent_dir
        print(pos_x, pos_y, curr_dir)
        print(self.env.unwrapped.front_pos)


def test_model():
    # Load the environment
    env = RandomMazeEnv(render_mode="human", size = 13)
    env = ImgObsWrapper(env)
    # env = DummyVecEnv([lambda: env])

    # Load the pre-trained model
    model = PPO.load("ppo_minigrid_logs/model_28260000_steps.zip")
    # model = BfsModel(env)

    # Reset the environment
    obs = env.reset()
    
    # Evaluate the model for a few steps
    for step in range(100):  # Adjust the step count as needed
        env.render()  # Render the environment
        pos_x, pos_y = env.unwrapped.agent_pos
        direction = env.unwrapped.agent_dir
        obs = env.unwrapped.gen_obs()
        action, _states = model.predict(obs["image"], deterministic=True)  # Get action from model
        # action, _ = bfsmodel.predict(obs)
        print(f"{pos_x=}, {pos_y=}, {direction=}, {action=}")
        
        obs, reward, trunc, term, info = env.step(action)  # Apply the action to the environment
        done = trunc or term
        print(f"{reward=}")
        # time.sleep(3)

        if done:
            print("Episode finished after {} steps".format(step + 1))
            obs = env.reset()  # Reset the environment

    env.close()  # Close the rendering window
    


# def aux_print_maze(maze):
#     res_str = ""
#     width, height = len(maze[0]), len(maze)
#     for i in range(width):
#         for j in range(height):
#             tmp = maze[i][j]
#             if tmp >= 10:
#                 res_str += f" {tmp}"
#             elif tmp > 0:
#                 res_str += f"  {tmp}"
#             elif tmp == 0:
#                 res_str += "  G"
#             else:
#                 res_str += "  X"
#         res_str += "\n"
#     return res_str

def aux_print_maze(maze):
    res_str = ""
    width, height = len(maze[0]), len(maze)

    for i in range(width):
        for j in range(height):
            tmp = maze[i][j]
            if tmp >= 10:
                res_str += f"  "
            elif tmp > 0:
                res_str += f"  "
            elif tmp == 0:
                res_str += "XG"
            else:
                res_str += "WG"
        res_str += "\n"
    res_str += "\n\n"

    for i in range(width):
        for j in range(height):
            tmp = maze[i][j]
            if tmp >= 10:
                res_str += f" {tmp}"
            elif tmp > 0:
                res_str += f"  {tmp}"
            elif tmp == 0:
                res_str += "  G"
            else:
                res_str += "  X"
        res_str += "\n"
    
    
    return res_str


def test_manual():
    env = RandomMazeEnv(render_mode="human", size = 13)
    gridstr = env.printGrid(init=True) # init=True is necessary to use minigrid2Prism
    print(gridstr)

    
    tmp_gridfile = 'tmp_grid.txt'

    with open(tmp_gridfile, 'w') as fp:
        fp.write(gridstr)

    print(aux_print_maze(env.distance_to_goal_grid))
    # import time
    # env.render()
    # time.sleep(100)
    # return

    
    # enable manual control for testing
    manual_control = ManualControl(env)    
    manual_control.start()


def generate_prism_guard_string(guard_name, conditions):
    res_str = f"formula {guard_name} = (false)"
    for cond in conditions:
        res_str += f" | (colAgent={cond[0]}&rowAgent={cond[1]}&viewAgent={cond[2]})"
    res_str += ";\n"
    return res_str

def generate_trace_input(env):
    res_str = ""
    grid = env.distance_to_goal_grid
    for pos_x in range(np.shape(grid)[0]):
        for pos_y in range(np.shape(grid)[1]):
            for direction in range(0,4):
                if grid[pos_y][pos_x] != -2:
                    res_str += f"colAgent={pos_x} & rowAgent={pos_y} & viewAgent={direction}\n"    
    return res_str


def get_prism_files(model, env):
    gridstr = env.printGrid(init=True) # init=True is necessary to use minigrid2Prism
    print(gridstr)

    
    
    tmp_gridfile = 'tmp_grid.txt'
    tmp_prismfile_mdp = 'tmp_grid_mdp.pm'
    tmp_prismfile_dtmc = 'tmp_grid_dtmc.pm'

    with open(tmp_gridfile, 'w') as fp:
        fp.write(gridstr)
    
    minigrid2prism = "./../Minigrid2PRISM/build/main"

    subprocess.run(
        [minigrid2prism, "-i", tmp_gridfile, "-o", tmp_prismfile_mdp],
        check=True,    # Raise an exception if the command fails
    )

    print(env.distance_to_goal_grid)
    print(np.shape(env.distance_to_goal_grid))

    print(aux_print_maze(env.distance_to_goal_grid))

    # env = ImgObsWrapper(env)
    # model_env = DummyVecEnv([lambda: model_env])

    move_forward = []
    turn_right = []
    turn_left = []
    do_nothing = []

    env.render()
    time.sleep(1)

    grid = env.distance_to_goal_grid
    for pos_x in range(np.shape(grid)[0]):
        for pos_y in range(np.shape(grid)[1]):
            for direction in range(0,4):
                if grid[pos_y][pos_x] != -2:
                    env.agent_pos = (pos_x, pos_y)

                    env.agent_dir = direction
                    # print(f"Outside: {env.agent_pos=}, {env.agent_dir=}")
                    obs = env.gen_obs()
                    action, _states = model.predict(obs["image"], deterministic=True)
                    if action == Actions.left:
                        turn_left.append((pos_x, pos_y, direction))
                    elif action == Actions.right:
                        turn_right.append((pos_x, pos_y, direction))
                    elif action == Actions.forward:
                        move_forward.append((pos_x, pos_y, direction))
                    else:
                        do_nothing.append((pos_x, pos_y, direction))

                    # print(pos_x, pos_y, direction, action)

    
    print(f"{len(move_forward)=}, {len(turn_left)=}, {len(turn_right)=}, {len(do_nothing)=}")

    print(generate_prism_guard_string("moveRight", turn_right))

    with open(tmp_prismfile_mdp, 'r') as fp:
        mdp_str_list = fp.readlines()


    dtmc_str_list = ['dtmc\n']

    for i in range(len(mdp_str_list)):
        if "module Agent" in mdp_str_list[i]:
            module_idx = i
            break
    
    for i in range(1, module_idx):
        dtmc_str_list.append(mdp_str_list[i])

    dtmc_str_list.append(generate_prism_guard_string("AgentTurnRight", turn_right))
    dtmc_str_list.append(generate_prism_guard_string("AgentTurnLeft", turn_left))
    dtmc_str_list.append(generate_prism_guard_string("AgentMoveForward", move_forward))
    dtmc_str_list.append(generate_prism_guard_string("AgentDoNothing", do_nothing))
    dtmc_str_list.append("\n")

    for i in range(module_idx, len(mdp_str_list)):
        line_str = mdp_str_list[i]
        if "[Agent_turn_right]" in line_str: 
            newline_str = line_str.split("->")[0] + " & AgentTurnRight ->" + line_str.split("->")[-1]
        elif "[Agent_turn_left]" in line_str:
            newline_str = line_str.split("->")[0] + " & AgentTurnLeft ->" + line_str.split("->")[-1]
        elif "[Agent_move_" in line_str:
            newline_str = line_str.split("->")[0] + " & AgentMoveForward ->" + line_str.split("->")[-1]
        elif "[Agent_done]" in line_str:
            newline_str = line_str.split("->")[0] + " & AgentDoNothing ->" + line_str.split("->")[-1]
        elif ("[Agent_pickup]" in line_str) or ("[Agent_drop]" in line_str) or ("[Agent_toggle]" in line_str):
            newline_str = ""
        else:
            newline_str = line_str
        dtmc_str_list.append(newline_str)
    
    with open(tmp_prismfile_dtmc, 'w') as fp:
        fp.write("".join(dtmc_str_list))

    res_str = generate_trace_input(env)
    with open('trace_input_adapted.txt', 'w') as fp:
        fp.write(res_str)
    return

#######################################





    for i in range(1,8):
        dtmc_str_list.append(mdp_str_list[i])

    dtmc_str_list.append(generate_prism_guard_string("AgentTurnRight", turn_right))
    dtmc_str_list.append(generate_prism_guard_string("AgentTurnLeft", turn_left))
    dtmc_str_list.append(generate_prism_guard_string("AgentMoveForward", turn_right))
    dtmc_str_list.append(generate_prism_guard_string("AgentDoNothing", do_nothing))

    # print(dtmc_str_list)

    for i in range(8,18):
        dtmc_str_list.append(mdp_str_list[i])
    line_str = mdp_str_list[18]
    newline_str = line_str.split("->")[0] + " & AgentTurnRight ->" + line_str.split("->")[-1]
    dtmc_str_list.append(newline_str)
    for i in [19, 20]:
        line_str = mdp_str_list[i]
        newline_str = line_str.split("->")[0] + " & AgentTurnLeft ->" + line_str.split("->")[-1]
        dtmc_str_list.append(newline_str)
    for i in [21, 22, 23, 24]:
        line_str = mdp_str_list[i]
        newline_str = line_str.split("->")[0] + " & AgentMoveForward ->" + line_str.split("->")[-1]
        dtmc_str_list.append(newline_str)
    line_str = mdp_str_list[28]
    newline_str = line_str.split("->")[0] + " & AgentDoNothing ->" + line_str.split("->")[-1]
    dtmc_str_list.append(newline_str)
    dtmc_str_list.append(mdp_str_list[29])

    with open(tmp_prismfile_dtmc, 'w') as fp:
        fp.write("".join(dtmc_str_list))









def main():
    # test_model()
    # return

    # print("start trainig...")
    # train()
    # print("end training...")
    # return


    # env = RandomMazeEnv(render_mode="human", size = 13)
    # model = PPO.load("ppo_minigrid_logs/ppo_minigrid.zip")
    # get_prism_files(model, env)
    # return


    env = RandomMazeEnv(render_mode="human", size = 13)
    model = PPO.load("ppo_minigrid_logs/model_28260000_steps.zip")
    # model = BfsModel(env)
    get_prism_files(model, env)
    return

    test_manual()
    return



    
if __name__ == "__main__":
    main()