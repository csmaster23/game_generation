import os
import random
from mechanics.Area_Movement import Area_Movement_Class
from mechanics.Hex_Grid_Movement import Hex_Grid_Movement_Class
from mechanics.Square_Grid_Movement import Square_Grid_Movement_Class

def generate_random_mechanics(num):
    print("Number of mechanics being generated: %d" % num)
    num = 1
    selected = []
    all_mechanics = []
    all_mechanics.append(Hex_Grid_Movement_Class())
    all_mechanics.append(Square_Grid_Movement_Class())
    for _ in num:
        selected.append(random.choice(all_mechanics))
    return selected





# Area Movement
    # Grid Movement
        # Hex Grid Movement
        # Square Grid Movement
    # Non-Grid Movement


