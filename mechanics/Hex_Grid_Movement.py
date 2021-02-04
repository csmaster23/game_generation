import numpy as np
from mechanics.Mechanic import Mechanic

class Hex_Grid_Movement_Class(Mechanic):
    def __init__(self, p):
        self.p = p