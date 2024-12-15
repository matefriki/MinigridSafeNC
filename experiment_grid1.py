import time, random, copy
import numpy as np
import pandas as pd
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
    fixed_balls = [(4,7), (1,2)]
    # fixed_balls = None
    fixed_goal = (5,7)
    # fixed_goal = None


    # fixed_slip = None

    env = MultiColorMazeEnv(render_mode="human", size = size,
                             fixed_maze=fixed_maze, fixed_balls_position=fixed_balls, fixed_goal_position=fixed_goal, goal_color="blue", fixed_slippery_tile_map=fixed_slip, probability_intended=0.3)
    
    # env = PurpleToEmptyWrapper(env)
    gridstr = env.unwrapped.printGrid(init=True) # init=True is necessary to use minigrid2Prism
    # print(gridstr)
    # manual_control = ManualControl(env)    
    # manual_control.start()
    # env.reset()
    # env.render()
    # time.sleep(10)

    # model = BfsModel(env)
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
    ball1 = (9,9)
    ball2 = (10,9)
    fixed_balls = [ball1, ball2]
    fixed_goal = generate_unique_tuple(fixed_balls, 2, 5)
    fixed_goal = (11,9)
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





def grid4():
    starttime = time.time()
    size = 13
    fixed_maze = [[0 for _ in range(size)] for _ in range(size)]

    # fixed_maze = None
    fixed_balls = [(5,6), (5,5), (6,5), (7,7), (7,6), (7,5), (6,7), (5,7)]
    # fixed_balls = [(5,6), (6,5), (7,6), (6,7), (5,5), (7,7)]
    # fixed_balls = None
    fixed_goal = (6,6)
    # fixed_goal = None
    n_slippery_tiles = 0
    fixed_slip = [[1 for _ in range(size)] for _ in range(size)]
    
    fixed_slip_copy = copy.deepcopy(fixed_slip)
    # fixed_slip = None
    fixed_agent = None

    env = MultiColorMazeEnv(render_mode="human", size = size,
                             fixed_maze=fixed_maze, fixed_balls_position=fixed_balls, fixed_goal_position=fixed_goal, goal_color="red", n_slippery_tiles=n_slippery_tiles, fixed_slippery_tile_map=copy.deepcopy(fixed_slip), fixed_agent_position=fixed_agent, probability_intended=0.9, probability_turn_intended=0.9)
    


    gridstr = env.printGrid(init=True) # init=True is necessary to use minigrid2Prism


    # print(gridstr)
    # env.reset()
    # env.render()
    # time.sleep(10)

    # model = BfsModel(env)
    # model = PPO.load("ppo_minigrid_logs/model_28260000_steps.zip")
    model = PPO.load("ppo_minigrid_logs/model_38600000_steps.zip")
    # test_env_model(env, model)
    
    iq = get_intention_quotients(env, model)
    print(iq)
    

    # env = MultiColorMazeEnv(render_mode="human", size = size,
    #                          fixed_maze=fixed_maze, fixed_balls_position=fixed_balls, fixed_goal_position=fixed_goal, goal_color="red", n_slippery_tiles=n_slippery_tiles, fixed_slippery_tile_map=fixed_slip, fixed_agent_position=fixed_agent, probability_intended=0.3, probability_turn_intended=0.3)

    env.fixed_slippery_tile_map = fixed_slip_copy

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
    fixed_balls = [(1,2), (8,2), (2,2), (9,2), (1,1)]
    # fixed_balls = None
    fixed_goal = (2,1)
    # fixed_goal = None
    n_slippery_tiles = 0
    fixed_slip = [[-1 for _ in range(size)] for _ in range(size)]
    for i in range(3,7):
        for j in range(3, size-3):
            fixed_slip[i][j] = 1
    fixed_slip_copy = copy.deepcopy(fixed_slip)
    # fixed_slip = None
    fixed_agent = None

    env = MultiColorMazeEnv(render_mode="human", size = size,
                             fixed_maze=fixed_maze, fixed_balls_position=fixed_balls, fixed_goal_position=fixed_goal, goal_color="red", n_slippery_tiles=n_slippery_tiles, fixed_slippery_tile_map=copy.deepcopy(fixed_slip), fixed_agent_position=fixed_agent, probability_intended=0.3, probability_turn_intended=0.3)
    


    gridstr = env.printGrid(init=True) # init=True is necessary to use minigrid2Prism


    # print(gridstr)
    # env.reset()
    # env.render()
    # time.sleep(10)

    # model = BfsModel(env)
    model = PPO.load("ppo_minigrid_logs/model_28260000_steps.zip")
    # model = PPO.load("ppo_minigrid_logs/model_38600000_steps.zip")
    # test_env_model(env, model)
    
    iq = get_intention_quotients(env, model)
    print(iq)
    

    # env = MultiColorMazeEnv(render_mode="human", size = size,
    #                          fixed_maze=fixed_maze, fixed_balls_position=fixed_balls, fixed_goal_position=fixed_goal, goal_color="red", n_slippery_tiles=n_slippery_tiles, fixed_slippery_tile_map=fixed_slip, fixed_agent_position=fixed_agent, probability_intended=0.3, probability_turn_intended=0.3)

    env.fixed_slippery_tile_map = fixed_slip_copy

    # enable manual control for testing
    manual_control = ManualControl(env)    
    manual_control.start()


def positions_at_distance(fixed_maze, ball_position, dist):
    rows, cols = len(fixed_maze), len(fixed_maze[0])
    close_positions = []

    # Iterate over all positions within a square of size (dist * 2 + 1)
    for di in range(-dist, dist + 1):
        dj = dist - abs(di)  # Ensure Manhattan distance is exactly 'dist'
        possible_positions = [
            (ball_position[0] + di, ball_position[1] + dj),
            (ball_position[0] + di, ball_position[1] - dj)
        ]

        for i, j in possible_positions:
            # Check if position is within bounds and valid
            if 0 < i < rows-1 and 0 < j < cols-1 and fixed_maze[i][j] == 0:
                close_positions.append((i, j))
    
    return close_positions


def random_triplet_sum(N: int):
    # Ensure N is between 0 and 100
    if not (0 <= N):
        raise ValueError("N must be between 0 and 100 inclusive.")
    
    x = random.randint(0, N)
    y = random.randint(0, N - x)
    z = N - (x + y)
    return x, y, z



def random_k_sum(N: int, k: int):
    """
    Generates k random integers that add up to N.

    Args:
    - N (int): The target sum.
    - k (int): The number of integers to generate.

    Returns:
    - List[int]: A list of k integers that add up to N.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    if N < 0:
        raise ValueError("N must be non-negative.")

    # Generate k-1 random breakpoints in the range [0, N]
    breakpoints = sorted(random.randint(0, N) for _ in range(k - 1))
    
    # Add 0 at the start and N at the end to form intervals
    breakpoints = [0] + breakpoints + [N]
    
    # Compute the differences between consecutive breakpoints
    result = [breakpoints[i + 1] - breakpoints[i] for i in range(k)]
    return result



def grid4_counterfactuals():
    size = 13
    fixed_maze = [[0 for _ in range(size)] for _ in range(size)]


    initial_balls = [(5,6), (5,5), (6,5), (7,7), (7,6), (7,5), (6,7), (5,7), (6,6)]
    # initial_goal = (6,6)
    n_slippery_tiles = 0
    fixed_slip = [[1 for _ in range(size)] for _ in range(size)]
    # fixed_slip = None

    close_to_ball_0 = {}
    close_to_ball_1 = {}
    close_to_goal = {}

    close_to_balls = [{} for k in range(len(initial_balls))]
    max_dist = 10


    for dist in range(0, max_dist):
        for idx in range(len(initial_balls)):
            close_to_balls[idx][dist] = positions_at_distance(fixed_maze, initial_balls[idx], dist)

    
    # for dist in range(0, max_dist):
        # close_to_ball_0[dist] = positions_at_distance(fixed_maze, initial_balls[0], dist)
        # close_to_ball_1[dist] = positions_at_distance(fixed_maze, initial_balls[1], dist)
        # close_to_goal[dist] = positions_at_distance(fixed_maze, initial_goal, dist)

    # print("Lens: ", len(close_to_ball_0[max_dist-1]), len(close_to_ball_1[max_dist-1]), len(close_to_goal[max_dist-1]))
    # return

    data = []

    already_seen = set()


    max_random_count = 100
    for tot_dist in range(0,max_dist):
        for _ in range(100):
            # print(tot_dist)
            Ds = random_k_sum(tot_dist, len(initial_balls))
            # dgoal = Ds[-1]

            random_counter = 0
            balls = [(0,0) for k in range(len(initial_balls))]

            idx_ball = 0
            conflict = False

            while (idx_ball < len(initial_balls)) and (random_counter < max_random_count):
                random_counter += 1
                balls[idx_ball] = close_to_balls[idx_ball][Ds[idx_ball]][np.random.randint(len(close_to_balls[idx_ball][Ds[idx_ball]]))]
                conflict = False
                for j in range(idx_ball):
                    conflict = conflict or (balls[j] == balls[idx_ball])
                if not conflict:
                    idx_ball += 1                

            #######

            # goal = balls[-1]
            if tuple(balls) in already_seen:
                random_counter = max_random_count
            if random_counter < max_random_count:
                already_seen.add(copy.deepcopy(tuple(balls)))
                
            
                fixed_balls = balls[:-1]
                fixed_goal = balls[-1]

                print(f"{fixed_balls=}, {fixed_goal=}")

                env = MultiColorMazeEnv(render_mode="human", size = size,
                                        fixed_maze=fixed_maze, fixed_balls_position=fixed_balls, fixed_goal_position=fixed_goal, goal_color="red", n_slippery_tiles=n_slippery_tiles, fixed_slippery_tile_map=fixed_slip, probability_intended=0.8)
                
                model = BfsModel(env)
                # model = PPO.load("ppo_minigrid_logs/model_28260000_steps.zip")
                # test_env_model(env, model)
            
                iq = get_intention_quotients(env, model)
                data.append((tot_dist, iq["red"], iq["blue"], iq["green"]))
                print(f"Dist: {tot_dist}, IQ: {iq}")
                # env.reset()
                # env.render()
                # time.sleep(3)
                # return
    
    df = pd.DataFrame(data, columns = ["dist", "red", "green", "blue"])
    df.to_csv('experimental_data/grid4_bfs.csv', index = False)





def grid2_counterfactuals():
    size = 13
    fixed_maze = [[0 for _ in range(size)] for _ in range(size)]

    for i in range(6,size):
        for j in range(4,9):
            fixed_maze[i][j] = 1

    initial_balls = [(9,9), (11,9)]
    initial_goal = (10,9)
    n_slippery_tiles = 0
    fixed_slip = [[1 for _ in range(size)] for _ in range(size)]
    # fixed_slip = None

    close_to_ball_0 = {}
    close_to_ball_1 = {}
    close_to_goal = {}

    max_dist = 10
    for dist in range(0, max_dist):
        close_to_ball_0[dist] = positions_at_distance(fixed_maze, initial_balls[0], dist)
        close_to_ball_1[dist] = positions_at_distance(fixed_maze, initial_balls[1], dist)
        close_to_goal[dist] = positions_at_distance(fixed_maze, initial_goal, dist)

    # print("Lens: ", len(close_to_ball_0[max_dist-1]), len(close_to_ball_1[max_dist-1]), len(close_to_goal[max_dist-1]))
    # return

    data = []

    max_random_count = 100
    for tot_dist in range(0,max_dist):
        for _ in range(10):
            print(tot_dist)
            d0, d1, d2 = random_k_sum(tot_dist, 3)
            random_counter = 0
            ball0 = close_to_ball_0[d0][np.random.randint(len(close_to_ball_0[d0]))]
            ball1 = close_to_ball_1[d1][np.random.randint(len(close_to_ball_1[d1]))]
            while ball0 == ball1 and random_counter < max_random_count:
                ball1 = close_to_ball_1[d1][np.random.randint(len(close_to_ball_1[d1]))]
                random_counter += 1
            goal = close_to_goal[d2][np.random.randint(len(close_to_goal[d2]))]
            while (goal == ball0 or goal == ball1) and (random_counter <  max_random_count):
                goal = close_to_goal[d2][np.random.randint(len(close_to_goal[d2]))]
                random_counter += 1
            
            if random_counter < max_random_count:
            
                fixed_balls = [ball0, ball1]
                fixed_goal = goal

                print(f"{fixed_balls=}, {fixed_goal=}")

                env = MultiColorMazeEnv(render_mode="human", size = size,
                                        fixed_maze=fixed_maze, fixed_balls_position=fixed_balls, fixed_goal_position=fixed_goal, goal_color="red", n_slippery_tiles=n_slippery_tiles, fixed_slippery_tile_map=fixed_slip, probability_intended=0.3)

                # model = BfsModel(env)
                model = PPO.load("ppo_minigrid_logs/model_28260000_steps.zip")
                # test_env_model(env, model)
            
                iq = get_intention_quotients(env, model)
                data.append((tot_dist, iq["red"], iq["blue"], iq["green"]))
                print(f"Dist: {tot_dist}, IQ: {iq}")
    
    df = pd.DataFrame(data, columns = ["dist", "red", "green", "blue"])
    df.to_csv('experimental_data/grid2_ppo_diff.csv', index = False)




def grid1_counterfactuals():
    starttime = time.time()
    size = 13
    fixed_maze = [[0 for _ in range(size)] for _ in range(size)]

    for i in range(3,7):
        for j in range(0,5):
            fixed_maze[i][j] = 1
        for j in range(-5,0):
            fixed_maze[i][j] = 1
    
    n_slippery_tiles = 0
    fixed_slip = [[-1 for _ in range(size)] for _ in range(size)]
    for i in range(3,7):
        for j in range(3, size-3):
            fixed_slip[i][j] = 1
    # fixed_slip = None


    initial_balls = [(2,2), (1,2)]
    initial_goal = (2,1)


    close_to_ball_0 = {}
    close_to_ball_1 = {}
    close_to_goal = {}

    max_dist = 10
    for dist in range(0, max_dist):
        close_to_ball_0[dist] = positions_at_distance(fixed_maze, initial_balls[0], dist)
        close_to_ball_1[dist] = positions_at_distance(fixed_maze, initial_balls[1], dist)
        close_to_goal[dist] = positions_at_distance(fixed_maze, initial_goal, dist)

    # print("Lens: ", len(close_to_ball_0[max_dist-1]), len(close_to_ball_1[max_dist-1]), len(close_to_goal[max_dist-1]))
    # return

    data = []

    max_random_count = 100
    for tot_dist in range(0,max_dist):
        for _ in range(10):
            print(tot_dist)
            d0, d1, d2 = random_k_sum(tot_dist,3)
            random_counter = 0
            ball0 = close_to_ball_0[d0][np.random.randint(len(close_to_ball_0[d0]))]
            ball1 = close_to_ball_1[d1][np.random.randint(len(close_to_ball_1[d1]))]
            while ball0 == ball1 and random_counter < max_random_count:
                ball1 = close_to_ball_1[d1][np.random.randint(len(close_to_ball_1[d1]))]
                random_counter += 1
            goal = close_to_goal[d2][np.random.randint(len(close_to_goal[d2]))]
            while (goal == ball0 or goal == ball1) and (random_counter <  max_random_count):
                goal = close_to_goal[d2][np.random.randint(len(close_to_goal[d2]))]
                random_counter += 1
            
            if random_counter < max_random_count:
            
                fixed_balls = [ball0, ball1]
                fixed_goal = goal

                print(f"{fixed_balls=}, {fixed_goal=}")

                env = MultiColorMazeEnv(render_mode="human", size = size,
                                        fixed_maze=fixed_maze, fixed_balls_position=fixed_balls, fixed_goal_position=fixed_goal, goal_color="red", n_slippery_tiles=n_slippery_tiles, fixed_slippery_tile_map=copy.deepcopy(fixed_slip), 
                                        fixed_agent_position=None, probability_intended=0.3,probability_turn_intended=0.3)
                
                # env = MultiColorMazeEnv(render_mode="human", size = size,
                #              fixed_maze=fixed_maze, fixed_balls_position=fixed_balls, fixed_goal_position=fixed_goal, goal_color="red", n_slippery_tiles=n_slippery_tiles, fixed_slippery_tile_map=copy.deepcopy(fixed_slip), fixed_agent_position=fixed_agent, probability_intended=0.3, probability_turn_intended=0.3)
                
                # env.reset()
                # env.render()
                # time.sleep(3)

                model = BfsModel(env)
                # model = PPO.load("ppo_minigrid_logs/model_28260000_steps.zip")
                # model = PPO.load("ppo_minigrid_logs/model_38600000_steps.zip")
                # test_env_model(env, model)
            
                iq = get_intention_quotients(env, model)
                data.append((tot_dist, iq["red"], iq["blue"], iq["green"]))
                print(f"Dist: {tot_dist}, IQ: {iq}")
    
    df = pd.DataFrame(data, columns = ["dist", "red", "green", "blue"])
    df.to_csv('experimental_data/grid1_bfs.csv', index = False)

def main():
    # grid4()
    grid4_counterfactuals()
    # grid1_counterfactuals()
    # grid2_counterfactuals()

if __name__ == "__main__":
    main()