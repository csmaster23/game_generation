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
                             1: "drag_to_capture",
                             2: "hop_to_capture",
                             3: "hop_over_capture",
                             4: "drag",
                             5: "hop"}

        self.reflect_patterns = self.p["reflect_patterns"]

        # These pattern symbols can be used for all the actions
        self.optional_pattern_symbols = {
            1: "NW", 2: "N", 3: "NE", 4: "W", 5: "E", 6: "SW", 7: "S", 8: "SE", 9: "*"
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

    def get_mechanic_dict(self):
        sq = {}
        # Key[level]            (Min, Max)
        sq["num_groups"] = (1, 1)  # num_group (how many of that mechanic there is)
        # Should record which group we are on
        sq["num_child_entities"] = (1, 6)  # num child entity types
        # Should record child entity type we are on
        sq["num_action_types"] = (1, max([x for x in self.action_types.keys() if type(x) is int]))  # num_action_types
        sq["num_patterns"] = (1, 4)  # num_patterns
        # Should record which pattern we are looking at
        sq["pattern_length"] = (1, 3)  # pattern_length
        sq["pattern_symbol"] = (1, max([x for x in self.optional_pattern_symbols.keys() if type(x) is int]))  # pattern_symbol
        sq["num_parent_entity_types"] = len(self.parent_entity_names)
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
