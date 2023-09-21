from minigrid.core.world_object import WorldObj
from minigrid.core.tasks import DoRandom, TaskManager
import numpy as np
import math


from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
    DIR_TO_VEC
)
from minigrid.utils.rendering import (
    fill_coords,
    point_in_triangle,
    rotate_fn,
)

class Adversary(WorldObj):
   def __init__(self, adversary_dir=1, color="blue", tasks=[DoRandom()]):
       super().__init__("adversary", color)
       self.adversary_dir = adversary_dir
       self.color = color
       self.task_manager = TaskManager(tasks)
       self.carrying = None

   def render(self, img):
       tri_fn = point_in_triangle(
           (0.12, 0.19),
           (0.87, 0.50),
           (0.12, 0.81),
       )

       # Rotate the agent based on its direction
       tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.adversary_dir)
       fill_coords(img, tri_fn, COLORS[self.color])

   def dir_vec(self):
       assert self.adversary_dir >= 0 and self.adversary_dir < 4
       return DIR_TO_VEC[self.adversary_dir]

   def can_overlap(self):
       return False

   def encode(self):
       return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.adversary_dir)
