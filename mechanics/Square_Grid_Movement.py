import numpy as np
from mechanics.Entity import Entity
from mechanics.Mechanic import Mechanic
from rl_code.Game import interpret_level

class Square_Grid_Movement_Class(Mechanic):
    def __init__(self, p=None):
        super().__init__(p)
        if p is None:
            self.p = {"grid_height" : 5,
                      "grid_width" : 5,
                      "reflect_patterns": True,
                      "default_action_types" : ["remove_captured_piece"], # replace_captured_piece
                      "parent_entity_names" : ["square", "reserve"]}
        self.mechanic_type = "Square-Grid Movement"
        self.grid_height = self.p['grid_height']
        self.grid_width = self.p['grid_width']
        self.parent_entity_names = {i+1: name for i, name in enumerate(self.p["parent_entity_names"])} # hand, draw_pile, discard_pile, playing_area
        self.child_entity_name = "square_movement_piece"
        self.action_types = {"default": self.p["default_action_types"],
                             # 1: "drag_to_capture",
                             1: "hop_to_capture",
                             # 3: "hop_over_capture",
                             # 4: "drag",
                             2: "hop"}


        self.parent_entities_to_action_types = {
            "square" : ['hop', 'hop_to_capture', 'remove_captured_piece'],
            "reserve" : ['replace_captured_piece']
        }

        self.reflect_patterns = self.p["reflect_patterns"]

        # These pattern symbols can be used for all the actions
        self.optional_pattern_symbols = {
            1: "NW", 2: "N", 3: "NE", 4: "W", 5: "E", 6: "SW", 7: "S", 8: "SE"
        }
        self.default_pattern_symbols = {
            1: "transfer"
        }

        self.all_pattern_symbols = {
            "drag_to_capture": self.optional_pattern_symbols,
            "hop_to_capture": self.optional_pattern_symbols,
            "hop_over_capture": self.optional_pattern_symbols,
            "drag": self.optional_pattern_symbols,
            "hop": self.optional_pattern_symbols,
            "remove_captured_piece": self.default_pattern_symbols,
            "replace_captured_piece": self.default_pattern_symbols,
        }

    def create_adjacency_matrix(self, target_indices, total_indices, pattern_symbol):
        len_target = len(target_indices)
        matrix = np.zeros((len(total_indices), len(total_indices)))
        grid_length = int(np.sqrt(len_target))

        if pattern_symbol in self.optional_pattern_symbols.values():
            # Created a system to quickly generate the adjacency matrices
            adjusted_target_indices = np.arange(len_target)

            # Figures out the north, east, south, and west indices
            index_dict = dict()
            index_dict["N"] = np.where(adjusted_target_indices // grid_length == 0)[0]
            index_dict["E"] = np.where(adjusted_target_indices % grid_length == grid_length - 1)[0]
            index_dict["W"] = np.where(adjusted_target_indices % grid_length == 0)[0]
            index_dict["S"] = np.where(adjusted_target_indices // grid_length == grid_length - 1)[0]

            for index in adjusted_target_indices:
                skip = False
                move_to_index = index
                for char in pattern_symbol:
                    if index in index_dict[char]:
                        skip = True
                if not skip:
                    if "N" in pattern_symbol:
                        move_to_index -= grid_length
                    if "S" in pattern_symbol:
                        move_to_index += grid_length
                    if "E" in pattern_symbol:
                        move_to_index += 1
                    if "W" in pattern_symbol:
                        move_to_index -= 1
                    real_index = target_indices[index]
                    real_move_to_index = target_indices[move_to_index]
                    matrix[real_index, real_move_to_index] = 1
        elif pattern_symbol in self.default_pattern_symbols.values():
            # Enable transfer to and from reserve
            # matrix[target_indices] = 1
            move_to_indices = []
            for index in total_indices:
                if index not in target_indices:
                    move_to_indices.append(index)
            matrix[target_indices, np.array(move_to_indices)] = 1
            matrix[np.diag_indices(grid_length)] = 0

        return matrix

    def create_adjacency_matrices(self, total_indices, target_indices, parent_name):
        parent_adjacency_matrices = dict()
        # First, get the parent type
        if "square" in parent_name:
            parent_type = "square"
        else:
            parent_type = "reserve"

        # Get the action types associated with that parent
        action_types = self.parent_entities_to_action_types[parent_type]

        # Loop through and get the adjaceny matrices for each action type and pattern_symbol
        for action_type in action_types:
            parent_adjacency_matrices[action_type] = dict()
            pattern_symbols = self.all_pattern_symbols[action_type]
            for pattern_symbol in pattern_symbols.values():
                parent_adjacency_matrices[action_type][pattern_symbol] = self.create_adjacency_matrix(target_indices, total_indices, pattern_symbol)
        return parent_adjacency_matrices

    def get_mechanic_dict(self):
        sq = {}
        # Key[level]            (Min, Max)
        sq["num_groups"] = (1, 1)  # num_group (how many of that mechanic there is)
        # Should record which group we are on
        sq["num_child_entities"] = (3, 4)  # num child entity types
        # Should record child entity type we are on
        sq["num_action_types"] = (1, max([x for x in self.action_types.keys() if type(x) is int]))  # num_action_types
        sq["num_patterns"] = (1, 4)  # num_patterns
        # Should record which pattern we are looking at
        sq["pattern_length"] = (1, 1)  # pattern_length
        sq["pattern_symbol"] = (1, max([x for x in self.optional_pattern_symbols.keys() if type(x) is int]))  # pattern_symbol
        sq["num_parent_entity_types"] = len(self.parent_entity_names)

        sq["parent_dup"] = (3, 3) # range to duplicate parent entities in
        # sq["parent_dup"] = {"square": (3, 3), "reserve": (1,1)}
        sq["child_dup"] = (2, 2) # range to duplicate children entities in
        return sq

    # def square_dict(self):
    #     sq = {}
    #     # Key[level]            (Min, Max)
    #     sq["num_groups"] = (1, 1)  # num_group (how many of that mechanic there is)
    #     # Should record which group we are on
    #     sq["num_child_entities"] = (1, 6)  # num child entity types
    #     # Should record child entity type we are on
    #     sq["num_action_types"] = (1, 4)  # num_action_types
    #     sq["num_patterns"] = (1, 2)  # num_patterns
    #     # Should record which pattern we are looking at
    #     sq["pattern_length"] = (1, 3)  # pattern_length
    #     sq["pattern_symbol"] = (1, 9)  # pattern_symbol
    #     sq["num_parent_entity_types"] = 1
    #     return sq
