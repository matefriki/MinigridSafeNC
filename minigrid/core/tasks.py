from abc import ABC
from typing import Iterable, List
try:
    from astar import find_path
except:
    print("Install with:")
    print("pip install git+https://github.com/jrialland/python-astar.git")
    raise Exception("Need to install astar")
import numpy.random

class Task(ABC):
    # returns a bool, true if task is completed, false otherwise
    def completed(self, pos, dir, carrying, env):
        pass
    # Returns the best action to solve this task
    def get_best_action(self, pos, dir, carrying, env):
        pass
    # returns a string representing the task
    def __repr__(self):
        pass


def get_plan(pos, dir, carrying, env, goal_pos):
    def neighbors_fnct_a_star(node):
        left = (node[0], node[1], node[3], -node[2])
        right = (node[0], node[1], -node[3], node[2])
        fwd_pos = node[0] + node[2], node[1] + node[3]
        forward_cell = env.grid.get(*fwd_pos)
        forward_background = env.grid.get_background(*fwd_pos)
        my_color = "red" if env.agent_pos == pos else env.grid.get(*pos).color
        background_color = forward_background.color if forward_background is not None else None

        forward_pos_open = forward_cell is None or forward_cell.can_overlap()
        forward_pos_not_agent = fwd_pos != env.agent_pos
        background_is_my_color_or_none_or_i_am_red = (forward_background is None or
                                                      (background_color == "lightblue" and my_color == "blue") or
                                                      (background_color == "lightgreen" and my_color == "green") or
                                                      (background_color == "lightgreen" and my_color == "purple") or # purple belongs to green region
                                                      my_color == "red") # red can do whatever

        if forward_pos_open and forward_pos_not_agent and  background_is_my_color_or_none_or_i_am_red:
            forward = (node[0] + node[2], node[1] + node[3], node[2], node[3])
            return forward, left, right
        else:
            return left, right

    plan = find_path(
        start=(pos[0], pos[1], dir[0], dir[1]),
        goal=(goal_pos[0], goal_pos[1], dir[0], dir[1]),
        neighbors_fnct=neighbors_fnct_a_star,
        reversePath=False,
        heuristic_cost_estimate_fnct=lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]),
        distance_between_fnct=lambda a, b: 1.0,
        is_goal_reached_fnct=lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1
    )
    return list(plan) if plan is not None else None


class GoTo(Task):
    def __init__(self, goal_position):
        self.goal_position = goal_position
        self.plan = None

    def completed(self, pos, dir, carrying, env):
        return pos == self.goal_position

    def get_best_action(self, pos, dir, carrying, env):

            # if farther than 1 unit away, Run A*
            if self.plan is None or len(self.plan) == 0:
                self.plan = get_plan(pos, dir, carrying, env, self.goal_position)

            # if we have a plan, but we are not in the state we should be, create new plan
            if self.plan is not None:
                current_state_maybe = self.plan.pop(0)
                if current_state_maybe[0] != pos[0] or \
                        current_state_maybe[1] != pos[1] or \
                        current_state_maybe[2] != dir[0] or \
                        current_state_maybe[3] != dir[1]:
                    self.plan = None # get_plan(pos, dir, carrying, env, self.goal_position)
                    return self.get_best_action(pos, dir, carrying, env)

            if self.plan is None or len(self.plan) <= 1:  # this will only happen if the agent is blocked in somehow
                return 6  # done, does nothing
            next_state = self.plan[0]

            # decide how to achieve next state
            if abs(next_state[0] - pos[0]) == 1 or abs(next_state[1] - pos[1]) == 1:
                return 2  # forward
            elif next_state[2] == dir[1] and next_state[3] == -dir[0]:
                return 0  # left
            elif next_state[2] == -dir[1] and next_state[3] == dir[0]:
                return 1  # right
            else:  # something went wrong such as bumping into other agent, replan
                self.plan = None
                return self.get_best_action(pos, dir, carrying, env)

    def __repr__(self):
        return "Task: Go to position {}".format(self.goal_position)


class PickUpObject(Task):
    def __init__(self, obj_position, obj):
        self.obj_position = obj_position
        self.obj = obj
        self.plan = None
    def completed(self, pos, dir, carrying, env):
        return carrying == self.obj
    def get_best_action(self, pos, dir, carrying, env):
        d_x = self.obj_position[0] - pos[0]
        d_y = self.obj_position[1] - pos[1]
        # if farther than 1 unit away, Run A*
        if abs(d_x) + abs(d_y) > 1:
            # if we have no plan, create new plan
            if self.plan is None or len(self.plan) == 0:
                self.plan = get_plan(pos, dir, carrying, env, self.obj_position)

            # if we have a plan, but we are not in the state we should be, create new plan
            if self.plan is not None:
                current_state_maybe = self.plan.pop(0)
                if current_state_maybe[0] != pos[0] or \
                        current_state_maybe[1] != pos[1] or \
                        current_state_maybe[2] != dir[0] or \
                        current_state_maybe[3] != dir[1]:
                    self.plan = get_plan(pos, dir, carrying, env, self.obj_position)

            if self.plan is None:  # this will only happen if the agent is blocked in somehow
                return 6  # done, does nothing
            next_state = self.plan[0]


            # decide how to achieve next state
            if abs(next_state[0] - pos[0]) == 1 or abs(next_state[1] - pos[1]) == 1:
                return 2 # forward
            elif next_state[2] == dir[1] and next_state[3] == -dir[0]:
                return 0 # left
            elif next_state[2] == -dir[1] and next_state[3] == dir[0]:
                return 1 # right
            else: # something went wrong such as bumping into other agent, replan
                self.plan = None
                return self.get_best_action(pos, dir, carrying, env)

        else:
            if d_x != dir[0] or d_y != dir[1]:
                return 0 # left
            else:
                return 3 # pickup

    def __repr__(self):
        return "Task: Pick up object at position {}".format(self.obj_position)


class PlaceObject(Task):
    def __init__(self, obj_position, obj):
        self.obj_position = obj_position
        self.obj = obj
        self.plan = None

    def completed(self, pos, dir, carrying, env):
        return env.grid.get(*self.obj_position) == self.obj and carrying is None
    def get_best_action(self, pos, dir, carrying, env):
        d_x = self.obj_position[0] - pos[0]
        d_y = self.obj_position[1] - pos[1]
        # if farther than 1 unit away, go to it
        if abs(d_x) + abs(d_y) > 1:
            # if we have no plan, create new plan
            if self.plan is None or len(self.plan) == 0:
                self.plan = get_plan(pos, dir, carrying, env, self.obj_position)

            # if we have a plan, but we are not in the state we should be, create new plan
            if self.plan is not None:
                current_state_maybe = self.plan.pop(0)
                if current_state_maybe[0] != pos[0] or \
                   current_state_maybe[1] != pos[1] or \
                   current_state_maybe[2] != dir[0] or \
                   current_state_maybe[3] != dir[1]:
                    self.plan = get_plan(pos, dir, carrying, env, self.obj_position)

            if self.plan is None: # this will only happen if the agent is blocked in somehow
                return 6 # done, does nothing
            next_state = self.plan[0]

            # decide how to achieve next state
            if abs(next_state[0] - pos[0]) == 1 or \
               abs(next_state[1] - pos[1]) == 1:
                return 2  # forward
            elif next_state[2] == dir[1] and next_state[3] == -dir[0]:
                return 0  # left
            elif next_state[2] == -dir[1] and next_state[3] == dir[0]:
                return 1  # right
            else:  # something went wrong such as bumping into other agent, replan
                self.plan = None
                return self.get_best_action(pos, dir, carrying, env)
        else:
            if d_x != dir[0] or d_y != dir[1]:
                return 0 # left
            else:
                return 4 # drop
    def __repr__(self):
        return "Task: Place object at position {}".format(self.obj_position)

class DoNothing(Task):
    def __init__(self):
        pass
    def completed(self, pos, dir, carrying, env):
        return False
    def get_best_action(self, pos, dir, carrying, env):
        return 6 # done
    def __repr__(self):
        return "Task: Do nothing"
class DoRandom(Task):
    def __init__(self):
        pass
    def completed(self, pos, dir, carrying, env):
        return False
    def get_best_action(self, pos, dir, carrying, env):
        return numpy.random.random_integers(0, 2, 1)# indexes between 0 and 2 are forward, left, right
    def __repr__(self):
        return "Task: Act randomly"
class TaskManager:
    def __init__(self, tasks:List[Task]):
        self.tasks = tasks

    def get_best_action(self, pos, dir, carrying, env):
        if len(self.tasks) == 0:
            raise Exception("Task empty")
        if self.tasks[0].completed(pos, dir, carrying, env):
            self.tasks.pop(0)
        best_action = self.tasks[0].get_best_action(pos, dir, carrying, env)
        return best_action