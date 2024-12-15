from __future__ import annotations

import numpy as np
import random, copy, subprocess, time, docker, os, re, json
from collections import deque
from gymnasium.core import Wrapper
from minigrid.core.constants import COLORS, COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import (
    Ball, Goal, Wall,
    SlipperyEast,
    SlipperySouth,
    SlipperyNorth,
    SlipperyWest
 )
from minigrid.minigrid_env import MiniGridEnv
from minigrid.manual_control import ManualControl
from minigrid.utils.rendering import fill_coords, point_in_circle
from minigrid.core.world_object import WorldObj
from minigrid.core.actions import Actions


DIRECTION_RIGHT = 0
DIRECTION_DOWN = 1
DIRECTION_LEFT = 2
DIRECTION_UP = 3

GOAL_COLORS = ["red", "green", "blue"]




class ColoredGoal(WorldObj):
    def __init__(self, color="red"):
        super().__init__("goal", color)

    def can_overlap(self):
        return True  # The agent can overlap with the goal

    def render(self, img):
        # Render the goal with the specified color
        c = COLORS[self.color]
        fill_coords(img, point_in_circle(0.5, 0.5, 0.3), c)


class PurpleToEmptyWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.purple_color_idx = COLOR_TO_IDX['purple']
        self.empty_object_idx = OBJECT_TO_IDX['empty']

    def observation(self, obs):
        # obs['image'] is a (view_size, view_size, 3) array
        img = obs['image']
        # Identify all positions where color is purple
        purple_mask = (img[:, :, 1] == self.purple_color_idx)
        # For these positions, replace the object channel with empty
        img[purple_mask, 0] = self.empty_object_idx
        return obs


class MultiColorMazeEnv(MiniGridEnv):
    def __init__(
        self,
        size=15,
        slippery_fraction = 0.2,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        probability_intended=8/9,
        probability_turn_intended=8/9,
        n_slippery_tiles = 0,
        fixed_maze = None,
        fixed_slippery_tile_map = None,
        fixed_agent_position = None,
        fixed_goal_position = None,
        fixed_balls_position = None,
        goal_color = "red",
        max_steps: int | None = None,
        env_type = "random",
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.slippery_fraction = slippery_fraction
        self.probability_intended = probability_intended
        self.probability_turn_intended = probability_turn_intended
        self.n_slippery_tiles = n_slippery_tiles

        self.fixed_maze = fixed_maze
        self.fixed_slippery_tile_map = fixed_slippery_tile_map
        self.fixed_agent_position = fixed_agent_position
        self.fixed_goal_position = fixed_goal_position
        self.fixed_balls_position = fixed_balls_position
        self.verify_fixed_elements(size)

        self.goal_color = goal_color



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
    

    def verify_fixed_elements(self, size):
        width, height = size, size

        if self.fixed_maze != None:
            assert len(self.fixed_maze) == width, "Fixed maze is of wrong width"
            assert len(self.fixed_maze[0]) == height, "Fixed maze is of wrong height"

        
        if self.fixed_slippery_tile_map != None:
            assert len(self.fixed_slippery_tile_map) == width, "Fixed maze is of wrong width"
            assert len(self.fixed_slippery_tile_map[0]) == height, "Fixed maze is of wrong height"

        if self.fixed_agent_position != None:
            x,y = self.fixed_agent_position
            assert 0 <= x and x < width, "Fixed agent position x out of bounds"
            assert 0 <= y and y < height, "Fixed agent position y out of bounds"
               
        if self.fixed_goal_position != None:
            x,y = self.fixed_goal_position
            assert 0 <= x and x < width, "Fixed goal position x out of bounds"
            assert 0 <= y and y < height, "Fixed goal position y out of bounds"

        if self.fixed_balls_position != None:
            # ball1, ball2 = self.fixed_balls_position
            for ball in self.fixed_balls_position:
                x, y = ball
                assert 0 <= x and x < width, "Fixed ball position x out of bounds"
                assert 0 <= y and y < height, "Fixed ball position y out of bounds"
        assert self.n_slippery_tiles < (height-1)*(width-1)


    


    def _gen_grid(self, width, height):
        if self.fixed_maze is None:
            # Generate a random maze
            maze = self.generate_maze_dfs(width,height)
        else:
            maze = copy.deepcopy(self.fixed_maze)

        # print(maze)

        # Initialize an empty grid
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)  # Create outer walls

        

        # Get (or randomly generate) slippery tile map/grid 
        if self.fixed_slippery_tile_map is None:
            slippery_tile_map = [[-1 for _ in range(width)] for _ in range(height)]
            n_slippery_tiles = self.n_slippery_tiles

            for i in range(n_slippery_tiles):
                direction_slippery = np.random.randint(0,4)
                pos_x = int(np.random.randint(1,width-1))
                pos_y = int(np.random.randint(1,height-1))
                slippery_tile_map[pos_x][pos_y] = direction_slippery
        else:
            slippery_tile_map = self.fixed_slippery_tile_map

        # Check that the grid does not overlap with set fixed positions
        if self.fixed_agent_position is not None:
            maze[self.fixed_agent_position[0]][self.fixed_agent_position[1]] = 0
            slippery_tile_map[self.fixed_agent_position[0]][self.fixed_agent_position[1]] = -1

        if self.fixed_goal_position is not None:
            maze[self.fixed_goal_position[0]][self.fixed_goal_position[1]] = 0
            slippery_tile_map[self.fixed_goal_position[0]][self.fixed_goal_position[1]] = -1
        if self.fixed_balls_position is not None:
            for ball in self.fixed_balls_position:
                maze[ball[0]][ball[1]] = 0
                slippery_tile_map[ball[0]][ball[1]] = -1

        # Populate the grid with walls
        for x in range(width):
            for y in range(height):
                if maze[x][y] == 1:
                    self.grid.set(x, y, Wall())


        
        colored_goal = ColoredGoal(self.goal_color)
        if self.fixed_goal_position is None:
            # Place the goal in another random accessible position
            self.goal_pos = self.place_obj(colored_goal, top=(1, 1), size=(width-2, height-2))
        else:
            # Place the goal in the fixed position
            # print("before goal")
            self.goal_pos = self.place_obj(colored_goal, top=self.fixed_goal_position, size=(1,1))
            # print("after goal")
        
        self.distance_to_goal_grid = self.generate_distance_to_goal(maze, self.goal_pos)
        
        ball_colors = GOAL_COLORS.copy()
        ball_colors.remove(self.goal_color)
        
        if self.fixed_balls_position is None:
            for col in ball_colors:
                self.place_obj(Ball(col), top=(1, 1), size=(width-2, height-2))            
        else:
            for idx in range(len(self.fixed_balls_position)):
                # print("before ball")
                self.place_obj(Ball(ball_colors[idx%(len(ball_colors))]), top=(self.fixed_balls_position[idx]), size=(1,1))
                # print("after ball")

        if self.fixed_agent_position is None:
            # print("Placing agent in random position")
            # Place the agent in a random accessible position
            self.place_agent(top=(1, 1), size=(width-2, height-2))
            # print(f"done_wiuth agetn, at pos {self.agent_pos=}")
        else:
            self.place_agent(top=self.fixed_agent_position, size = (1,1))

        # Populate the grid with slippery tiles
        for pos_x in range(len(slippery_tile_map)):
            for pos_y in range(len(slippery_tile_map[0])):
                cell = self.grid.get(pos_x, pos_y)  # Get the object at the grid position
        # Check if the cell is not a wall or is empty
                # if cell is None or not isinstance(cell, Wall):
                if cell is None:
                # if self.grid.get(pos_x, pos_y) is None:
                    # if slippery_tile_map[pos_x][pos_y] != -1:
                    #     print(f"Free at {pos_x=}, {pos_y=}")
                    # only place slippery tiles in empty tiles
                    if slippery_tile_map[pos_x][pos_y] == DIRECTION_UP:
                        self.grid.horz_wall(pos_x, pos_y, 1, SlipperyNorth(probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
                    if slippery_tile_map[pos_x][pos_y] == DIRECTION_DOWN:
                        self.grid.horz_wall(pos_x, pos_y, 1, SlipperySouth(probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
                    if slippery_tile_map[pos_x][pos_y] == DIRECTION_RIGHT:
                        self.grid.horz_wall(pos_x, pos_y, 1, SlipperyEast(probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))
                    if slippery_tile_map[pos_x][pos_y] == DIRECTION_LEFT:
                        self.grid.horz_wall(pos_x, pos_y, 1, SlipperyWest(probability_intended=self.probability_intended, probability_turn_intended=self.probability_turn_intended))

        
        

    def step(self, action):
        
        pos_x, pos_y = self.agent_pos[0], self.agent_pos[1]
        # print(f"before: {self.agent_pos=}")
        prev_distance = self.distance_to_goal_grid[pos_x][pos_y]
        
        obs, reward, done, truncated, info = super().step(action)

        pos_x, pos_y = self.agent_pos[0], self.agent_pos[1]
        curr_distance = self.distance_to_goal_grid[pos_x][pos_y]

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
        # The maze is generated for a map with half width and height, 
        # so that in the end the paths are wider

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
                    maze[nx][ny] = 0
                    carve_passages_from(nx, ny)
        
        # Start carving from a random cell
        start_x, start_y = random.randrange(1, width, 2), random.randrange(1, height, 2)
        maze[start_x][start_y] = 0
        carve_passages_from(start_x, start_y)

        # once the small maze has been built, 
        # maze_large is the same maze doubling every tile
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

        return maze_large

    
    def generate_distance_to_goal(self, maze, goal):
        # Distance to goal generated with a standard BFS
        distance_maze = copy.deepcopy(maze)
        goal_x, goal_y = goal
        width, height = len(distance_maze[0]), len(distance_maze)
        for i in range(width):
            for j in range(height):
                if distance_maze[i][j] == 1:
                    distance_maze[i][j] = -2 # wall
                else:
                    distance_maze[i][j] = -1 # not visited
        distance_maze[goal_x][goal_y] = 0

        queue = deque([(goal_x, goal_y)])

        while queue:
            node = queue.popleft()
            node_x, node_y = node
            current_dist = distance_maze[node_x][node_y]
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dx, dy in directions:
                nx, ny = node_x + dx, node_y + dy  # Move one cell at a time
                if 0 <= nx < width and 0 <= ny < height and distance_maze[nx][ny] == -1:
                    distance_maze[nx][ny] = current_dist + 1
                    queue.append((nx,ny))
        return distance_maze
            



                
    
    def printGrid(self, init=False):
        grid = super().printGrid(init)

        properties_str = ""

        properties_str += F"ProbTurnIntended:{self.probability_turn_intended}\n"
        properties_str += F"ProbForwardIntended:{self.probability_intended}\n"

        return grid + properties_str
    
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        return obs


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
        curr_distance = self.env.unwrapped.distance_to_goal_grid[pos_x][pos_y]
        fwd_distance = self.env.unwrapped.distance_to_goal_grid[fwd_pos_x][fwd_pos_y]
        if fwd_distance >= curr_distance or fwd_distance == -2 :
            return 0, 0
        return 2, 2
    
def test_env_model(env, model):
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



def generate_prism_guard_string_for_actions(guard_name, conditions):
    res_str = f"formula {guard_name} = (false)"
    for cond in conditions:
        res_str += f" | (colAgent={cond[0]}&rowAgent={cond[1]}&viewAgent={cond[2]})"
    res_str += ";\n"
    return res_str

def generate_prism_guard_string_for_goals(guard_name, conditions):
    res_str = f"formula {guard_name} = (false)"
    for cond in conditions:
        res_str += f" | (colAgent={cond[0]}&rowAgent={cond[1]})"
    res_str += ";\n"
    return res_str

def generate_trace_input(env):
    res_str = ""
    grid = env.distance_to_goal_grid
    for pos_x in range(np.shape(grid)[0]):
        for pos_y in range(np.shape(grid)[1]):
            for direction in range(0,4):
                if grid[pos_x][pos_y] != -2:
                    res_str += f"colAgent={pos_x} & rowAgent={pos_y} & viewAgent={direction}\n"    
    return res_str

def read_prob_average_from_mc_result(resfile):
    values = []
    value_pattern = re.compile(r'"v"\s*:\s*([0-1](?:\.\d+)?)')
    with open(resfile, "r") as fp:
        for line in fp:
            match = value_pattern.search(line)
            if match:
                values.append(float(match.group(1)))
    return sum(values) / len(values)


def load_values(file_path):
    values = {}
    with open(file_path, 'r') as file:
        asdict = json.load(file)

    for i in range(len(asdict)):
        colagent = asdict[i]['s']['colAgent']
        rowagent = asdict[i]['s']['rowAgent']
        viewagent = asdict[i]['s']['viewAgent']

        key = (colagent, rowagent, viewagent)
        values[key] = asdict[i]['v']
    return values

def compute_iq_from_files(filemin, filemid, filemax):    
    values_min = load_values(filemin)
    values_max = load_values(filemax)
    values_mid = load_values(filemid)

    shared_keys = set(values_mid.keys())
    shared_keys &= set(values_max.keys()) # Intersect to find shared keys
    shared_keys &= set(values_min.keys()) # Intersect to find shared keys

    values_min_masked = [values_min[key] for key in shared_keys]
    values_mid_masked = [values_mid[key] for key in shared_keys]
    values_max_masked = [values_max[key] for key in shared_keys]

    # print(f"{len(shared_keys)=}")

    pmin = sum(values_min_masked)
    pmax = sum(values_max_masked)
    pmid = sum(values_mid_masked)

    return (pmid - pmin)/(pmax - pmin)



def mask_purple(obs):
    purple_color_idx = COLOR_TO_IDX['purple']
    empty_object_idx = OBJECT_TO_IDX['empty']
    empty_color_idx = COLOR_TO_IDX['black']
    # img = obs['image']
    # Identify all positions where color is purple
    purple_mask = (obs[:, :, 1] == purple_color_idx)
    # For these positions, replace the object channel with empty
    obs[purple_mask, 0] = empty_object_idx
    obs[purple_mask, 1] = empty_color_idx
    return obs

def get_intention_quotients(env, model):


    gridstr = env.printGrid(init=True) # init=True is necessary to use minigrid2Prism
    # print(gridstr)
    
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

    # return 0

    # print(env.distance_to_goal_grid)
    # print(np.shape(env.distance_to_goal_grid))

    # print(aux_print_maze(env.distance_to_goal_grid))

    # env = ImgObsWrapper(env)
    # model_env = DummyVecEnv([lambda: model_env])

    move_forward = []
    turn_right = []
    turn_left = []
    do_nothing = []
    red_cells = []
    green_cells = []
    blue_cells = []

    # env.render()
    # time.sleep(1)
    
    fixed_slip = env.fixed_slippery_tile_map
    for i in range(len(fixed_slip)):
        for j in range(len(fixed_slip[0])):
            fixed_slip[i][j] = -1
    env.fixed_slippery_tile_map = fixed_slip
    env.render_mode = "none"
    env.reset()
    env.render_mode = "human"

    grid = env.distance_to_goal_grid
    for pos_x in range(np.shape(grid)[0]):
        for pos_y in range(np.shape(grid)[1]):
            obj = env.grid.get(pos_x, pos_y)
            if obj is not None:
                if obj.color == "red":
                    red_cells.append((pos_x, pos_y))
                elif obj.color == "green":
                    green_cells.append((pos_x, pos_y))
                elif obj.color == "blue":
                    blue_cells.append((pos_x, pos_y))

            for direction in range(0,4):
                if grid[pos_x][pos_y] != -2:
                    env.agent_pos = (pos_x, pos_y)

                    env.agent_dir = direction
                    # print(f"Outside: {env.agent_pos=}, {env.agent_dir=}")
                    obs = env.gen_obs()
                    action, _states = model.predict(mask_purple(obs["image"]), deterministic=True)
                    if action == Actions.left:
                        turn_left.append((pos_x, pos_y, direction))
                    elif action == Actions.right:
                        turn_right.append((pos_x, pos_y, direction))
                    elif action == Actions.forward:
                        move_forward.append((pos_x, pos_y, direction))
                    else:
                        do_nothing.append((pos_x, pos_y, direction))

                    # print(pos_x, pos_y, direction, action)

    
    # print(f"{len(move_forward)=}, {len(turn_left)=}, {len(turn_right)=}, {len(do_nothing)=}")

    # print(generate_prism_guard_string("moveRight", turn_right))

    with open(tmp_prismfile_mdp, 'r') as fp:
        mdp_str_original_list = fp.readlines()


    dtmc_str_list = ['dtmc\n']
    mdp_str_list = ['mdp\n']

    for i in range(len(mdp_str_original_list)):
        if "module Agent" in mdp_str_original_list[i]:
            module_idx = i
            break
    
    for i in range(1, module_idx):
        dtmc_str_list.append(mdp_str_original_list[i])
        mdp_str_list.append(mdp_str_original_list[i])

    dtmc_str_list.append(generate_prism_guard_string_for_actions("AgentTurnRight", turn_right))
    dtmc_str_list.append(generate_prism_guard_string_for_actions("AgentTurnLeft", turn_left))
    dtmc_str_list.append(generate_prism_guard_string_for_actions("AgentMoveForward", move_forward))
    dtmc_str_list.append(generate_prism_guard_string_for_actions("AgentDoNothing", do_nothing))
    dtmc_str_list.append("\n")
    dtmc_str_list.append(generate_prism_guard_string_for_goals("AgentOnRed", red_cells))
    dtmc_str_list.append(generate_prism_guard_string_for_goals("AgentOnGreen", green_cells))
    dtmc_str_list.append(generate_prism_guard_string_for_goals("AgentOnBlue", blue_cells))
    dtmc_str_list.append("\n")

    mdp_str_list.append(generate_prism_guard_string_for_goals("AgentOnRed", red_cells))
    mdp_str_list.append(generate_prism_guard_string_for_goals("AgentOnGreen", green_cells))
    mdp_str_list.append(generate_prism_guard_string_for_goals("AgentOnBlue", blue_cells))
    mdp_str_list.append("\n")

    for i in range(module_idx, len(mdp_str_original_list)):
        line_str = mdp_str_original_list[i]
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
        mdp_str_list.append(line_str)

    with open(tmp_prismfile_dtmc, 'w') as fp:
        fp.write("".join(dtmc_str_list))
    
    with open(tmp_prismfile_mdp, 'w') as fp:
        fp.write("".join(mdp_str_list))

    res_str = generate_trace_input(env)
    tmp_traceinputfile = 'tmp_trace_input_adapted.txt'
    
    with open(tmp_traceinputfile, 'w') as fp:
        fp.write(res_str)

    tmp_mc_resultsfile_dtmc = "tmp_mc_result_dtmc.txt"
    tmp_mc_resultsfile_mdpmax = "tmp_mc_result_mdp.txt"
    tmp_mc_resultsfile_mdpmin = "1tmp_mc_result_mdp.txt"

    intention_quotients = {}

    client = docker.from_env()

    container = client.containers.run(
        "lposch/tempest-devel-traces:latest",
        command="sleep infinity",  # Keep the container alive
        volumes={os.getcwd(): {'bind': '/mnt/vol1', 'mode': 'rw'}},
        working_dir="/mnt/vol1",
        detach=True
    )

    for col in ["red", "green", "blue"]:
        # DTMC
        command = f"storm --prism {tmp_prismfile_dtmc} --prop 'P=? [F AgentOn{col.capitalize()}]' --trace-input {tmp_traceinputfile} --exportresult {tmp_mc_resultsfile_dtmc} --buildstateval"
        exec_result = container.exec_run(command, stderr=True)
        # dtmc_prob_averages[col] = read_prob_average_from_mc_result(f"{tmp_mc_resultsfile_dtmc}")

        # MDP max and # MDP min
        command = f"storm --prism {tmp_prismfile_mdp} --prop 'Pmax=? [F AgentOn{col.capitalize()}]; Pmin=? [F AgentOn{col.capitalize()}]' --trace-input {tmp_traceinputfile} --exportresult {tmp_mc_resultsfile_mdpmax} --buildstateval"
        exec_result = container.exec_run(command, stderr=True)
        # mdp_max_prob_averages[col] = read_prob_average_from_mc_result(f"{tmp_mc_resultsfile}_mdpmax.txt")

        intention_quotients[col] = compute_iq_from_files(tmp_mc_resultsfile_mdpmin, tmp_mc_resultsfile_dtmc, tmp_mc_resultsfile_mdpmax)



    # container.stop()
    container.kill() # killing direclty without stop, because stop takes too long
    container.remove()


    return intention_quotients


    




    

    
    

if __name__ == "__main__":
    size = 13
    fixed_maze = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        pass
        # fixed_maze[0][i] = 1
        # fixed_maze[i][0] = 1
        # fixed_maze[i][-1] = 1
        # fixed_maze[-1][i] = 1
    fixed_maze[2][5] = 1
    fixed_maze[2][6] = 1
    fixed_maze[2][7] = 1
    fixed_maze[2][8] = 1
    
    fixed_agent = (4,5)
    fixed_balls = [(4,6), (4,7)]
    fixed_goal = (4,8)
    fixed_slip = [[-1 for _ in range(size)] for _ in range(size)]
    fixed_slip[1][5] = 1
    fixed_slip[1][6] = 1
    fixed_slip[1][7] = 1
    fixed_slip[1][8] = 1
    fixed_slip = None
    fixed_maze = None
    # fixed_goal = None
    # fixed_balls = None
    # fixed_agent = None
    env = MultiColorMazeEnv(render_mode="human", size = size,
                             fixed_maze=fixed_maze, fixed_agent_position=fixed_agent, fixed_balls_position=fixed_balls, fixed_goal_position=fixed_goal, goal_color="green", n_slippery_tiles=10, fixed_slippery_tile_map=fixed_slip)
    gridstr = env.printGrid(init=True) # init=True is necessary to use minigrid2Prism
    print(gridstr)

    get_intention_quotients(env, BfsModel(env))

    print("Done")
 
    # enable manual control for testing
    # manual_control = ManualControl(env)    
    # manual_control.start()