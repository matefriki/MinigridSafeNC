import time
import numpy as np
from multicolor_maze import MultiColorMazeEnv, BfsModel, PurpleToEmptyWrapper
from minigrid.manual_control import ManualControl
from multicolor_maze import get_intention_quotients, test_env_model
from stable_baselines3 import PPO

def generate_unique_tuple(existing_tuples, range_start, range_end):
    while True:
        new_tuple = (np.random.randint(range_start, range_end), np.random.randint(range_start, range_end))
        if new_tuple not in existing_tuples:
            return new_tuple


def grid3():
    size = 13
    fixed_maze = [[0 for _ in range(size)] for _ in range(size)]

    focus_point = [4,7]
    maindig = focus_point[0] - focus_point[1]
    invdig = focus_point[0] + focus_point[1]

    fixed_slip = [[-1 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        for j in range(size):
            if i-j < maindig:
                if i+j > invdig:
                    fixed_slip[i][j] = 3
                elif i+j < invdig:
                    fixed_slip[i][j] = 0
            elif i-j > maindig:
                if i+j > invdig:
                    fixed_slip[i][j] = 2
                elif i+j < invdig:
                    fixed_slip[i][j] = 1


    # fixed_maze = None
    fixed_balls = [(4,7), (5,7)]
    # fixed_balls = None
    fixed_goal = (2,1)
    # fixed_goal = None


    # fixed_slip = None

    env = MultiColorMazeEnv(render_mode="human", size = size,
                             fixed_maze=fixed_maze, fixed_balls_position=fixed_balls, fixed_goal_position=fixed_goal, goal_color="green", fixed_slippery_tile_map=fixed_slip, probability_intended=0.3)
    
    # env = PurpleToEmptyWrapper(env)
    gridstr = env.unwrapped.printGrid(init=True) # init=True is necessary to use minigrid2Prism
    # print(gridstr)
    # manual_control = ManualControl(env)    
    # manual_control.start()
    # env.reset()
    # env.render()
    # time.sleep(10)

    model = BfsModel(env)
    model = PPO.load("ppo_minigrid_logs/model_28260000_steps.zip")
    # env = PurpleToEmptyWrapper(env)
    # test_env_model(env, model)
    iq = get_intention_quotients(env, model)
    print(iq)
    # env.render()
    # time.sleep(3)
    
 
    # enable manual control for testing
    manual_control = ManualControl(env)    
    manual_control.start()


def grid2():
    size = 13
    fixed_maze = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        pass
        # fixed_maze[0][i] = 1
        # fixed_maze[i][0] = 1
        # fixed_maze[i][-1] = 1
        # fixed_maze[-1][i] = 1
    for i in range(6,size):
        for j in range(4,9):
            fixed_maze[i][j] = 1


    ball1 = generate_unique_tuple([], 2,5)
    ball2 = generate_unique_tuple([ball1], 2, 5)
    ball1 = (9,11)
    ball2 = (10,11)
    fixed_balls = [ball1, ball2]
    fixed_goal = generate_unique_tuple(fixed_balls, 2, 5)
    fixed_goal = (11,11)
    # fixed_maze = None
    # fixed_balls = [(np.random.randint(2,5),np.random.randint(2,5)), (np.random.randint(2,5),np.random.randint(2,5))]
    # fixed_balls = None
    # fixed_goal = (np.random.randint(2,5),np.random.randint(2,5))
    # fixed_goal = None
    n_slippery_tiles = 0
    fixed_slip = [[1 for _ in range(size)] for _ in range(size)]
    # fixed_slip = None
    print(f"{fixed_balls=}, {fixed_goal=}")
    env = MultiColorMazeEnv(render_mode="human", size = size,
                             fixed_maze=fixed_maze, fixed_balls_position=fixed_balls, fixed_goal_position=fixed_goal, goal_color="red",
                            n_slippery_tiles=n_slippery_tiles, fixed_slippery_tile_map=fixed_slip, probability_intended=0.6)
    gridstr = env.printGrid(init=True) # init=True is necessary to use minigrid2Prism
    # print(gridstr)

    model = BfsModel(env)
    # test_env_model(env, model)
    iq = get_intention_quotients(env, model)
    print(iq)

    # enable manual control for testing
    manual_control = ManualControl(env)    
    manual_control.start()


def grid1():
    starttime = time.time()
    size = 13
    fixed_maze = [[0 for _ in range(size)] for _ in range(size)]

    for i in range(3,7):
        for j in range(0,5):
            fixed_maze[i][j] = 1
        for j in range(-5,0):
            fixed_maze[i][j] = 1
    # fixed_maze = None
    fixed_balls = [(4,7), (5,7)]
    # fixed_balls = None
    fixed_goal = (2,1)
    # fixed_goal = None
    n_slippery_tiles = 0
    fixed_slip = [[-1 for _ in range(size)] for _ in range(size)]
    for i in range(3,7):
        for j in range(5, size-5):
            fixed_slip[i][j] = 1
    # fixed_slip = None

    env = MultiColorMazeEnv(render_mode="human", size = size,
                             fixed_maze=fixed_maze, fixed_balls_position=fixed_balls, fixed_goal_position=fixed_goal, goal_color="green", n_slippery_tiles=n_slippery_tiles, fixed_slippery_tile_map=fixed_slip, probability_intended=0.3)

    gridstr = env.printGrid(init=True) # init=True is necessary to use minigrid2Prism
    
    
    # print(gridstr)
    # env.reset()
    # env.render()
    # time.sleep(10)

    model = BfsModel(env)
    # test_env_model(env, model)
    
    iq = get_intention_quotients(env, model)
    print(iq)

    # return

    # enable manual control for testing
    manual_control = ManualControl(env)    
    manual_control.start()
 

def main():
    grid3()

if __name__ == "__main__":
    main()