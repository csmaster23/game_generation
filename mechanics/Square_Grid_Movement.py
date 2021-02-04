import numpy as np
from mechanics.Entity import Entity
from mechanics.Mechanic import Mechanic

class Square_Grid_Movement_Class(Mechanic):
    def __init__(self, p):
        self.p = p
        self.mechanic_type = p['mechanic_type']
        self.grid_height = p['grid_height']
        self.grid_width = p['grid_width']


    def create_entity(self):
        entities = []
        for i in range(self.width * self.grid_height):
            entities.append( Entity( self.p, self.mechanic_type ) )
        return entities
