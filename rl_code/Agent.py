from mechanics.Entity import Entity
import numpy as np

class RandomAgent():
    def __init__(self):
        pass

    def take_action(self, min_choice, max_choice):
        return np.random.randint(min_choice, max_choice+1)

class Agent():
    def __init__(self, p):
        self.p = p


        # "Square-Grid Movement"
        # "Static Capture"

    def generate_entities(self):
        '''
        params:
        :return: list of generic entities
        '''
        print("\n--- Top of Generate Entities ---")
        print("Mechanic List: %s" % str(self.p['mechanics']))
        sq = "Square-Grid Movement"
        if sq in self.p['mechanics']:
            h_w_tuple = self.choose_grid_size()
            grids = self.create_grids(h_w_tuple, sq)
            for grid in grids:
                print("Grid Id: %s" % grid.id)
                num_pieces = self.choose_num_pieces()
        print("Done...")
        return []

    def choose_grid_size(self):
        return (5,5)
    def choose_num_pieces(self):
        return 1
    def create_grids(self, h_w_tuple, type_):
        grids = []
        for h in range(h_w_tuple[0]):
            for w in range(h_w_tuple[1]):
                id_ = str(h) + " " + str(w)
                grids.append( Entity( type_, id_, self.p ) )
        return grids