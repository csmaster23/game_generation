import uuid

def square_dict(self):
    sq = {}
    # Key[level]            (Min, Max)
    sq[1] = (1, 1)  # num_grids
    sq[2] = (1, 1)  # square_types
    # Should record square type we are on
    sq[4] = (1, 6)  # piece_types
    # Should record piece type we are on
    sq[5] = (1, 3)  # num_action_types
    sq[6] = (1, 4)  # selected_action_type. Needs to be masked out
    sq[7] = (1, 2)  # num_patterns
    # Should record which pattern we are looking at
    sq[8] = (1, 3)  # pattern_length
    sq[9] = (1, 9)  # pattern_symbols
    return sq

class Entity():
    def __init__(self, entity_type, sub_type, parent, id=None, p=None):
        self.p = p
        self.entity_type = entity_type # string e.x. Square_Grid
        self.parents = [parent] # Types that the entity can move across
        self.entity_type_list = [entity_type + '_' + str(sub_type)]
        if id is None:
            self.id = uuid.uuid1()
        else:
            self.id = id
        self.my_store_types = []
        self.my_storage_location = 0
        self.my_stored_ids = []
