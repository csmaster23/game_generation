import numpy as np
from mechanics.Entity import Entity
from mechanics.Mechanic import Mechanic
from rl_code.Game import interpret_level

class Deck_of_Cards_Class(Mechanic):
    def __init__(self, p=None):
        super().__init__(p)
        if p is None:
            self.p = {"num_cards" : 52,
                      "parent_entity_names": ["draw", "discard", "playing_area", "hand"]
                      "reflect_patterns": False,
                      "draw_discard": True,
                      "player_hand": False}
        self.mechanic_type = "Deck-of-Cards"
        self.num_cards = self.p['num_cards']
        self.parent_entity_names = {[i]: }
        self.child_entity_name = "card"
        self.action_types = {"default": ["remove_captured_piece"],
                             1: "drag_to_capture",
                             2: "hop_to_capture",
                             3: "hop_over_capture",
                             4: "drag",
                             5: "hop"}
        if self.p["replace_captured_pieces"]:
            self.action_types["default"].append("replace_captured_piece")
        self.reflect_patterns = self.p["reflect_patterns"]

        # These pattern symbols can be used for all the actions
        self.pattern_symbols1 = {
            1: "black", 2: "red", 3: "blue", 4: "green", 5: "1", 6: "2", 7: "3", 8: "4", 9: "5", 10: "6", 11: "7",
            12: "8", 13: "9", 14: "10", 15: "11", 16: "12", 17: "13"
        }
        self.pattern_symbols2 = {
            1: "transfer"
        }
        self.all_pattern_symbols = {"remove_captured_piece" : self.pattern_symbols2,
                             "drag_to_capture": self.pattern_symbols1,
                             "hop_to_capture": self.pattern_symbols1,
                             "hop_over_capture": self.pattern_symbols1,
                             "drag": self.pattern_symbols1,
                             "hop": self.pattern_symbols1,
                             "replace_captured_piece": self.pattern_symbols2}


    def square_dict(self):
        sq = {}
        # Key[level]            (Min, Max)
        sq["num_groups"] = (1, max([x for x in self.parent_entity_names.keys() if type(x) is int]))  # num_group (how many of that mechanic there is)
        # Should record which group we are on
        sq["num_child_entities"] = (1, 6)  # num child entity types
        # Should record child entity type we are on
        sq["num_action_types"] = (1, max([x for x in self.action_types.keys() if type(x) is int]))  # num_action_types
        sq["num_patterns"] = (1, 4)  # num_patterns
        # Should record which pattern we are looking at
        sq["pattern_length"] = (1, 3)  # pattern_length
        sq["pattern_symbol"] = (1, max([x for x in self.pattern_symbols1.keys() if type(x) is int]))  # pattern_symbol
        sq["num_parent_entity_types"] = 2
        return sq

    interpret_level = {
        0: "mechanic_num",
        1: "num_groups",
        2: "selected_group",
        3: "selected_parent_entity",  # Defaults to 0
        4: "num_child_entities",
        5: "selected_child_entity",
        6: "num_action_types",
        7: "selected_action_type",
        8: "num_patterns",
        9: "selected_pattern",
        10: "pattern_length",
        11: "pattern_symbol"
    }

    def add_to_entity(self, entity, tree_trajectory):
        [1, 1, 1, 0, 1, 1, 3, 1, 1, 1, 3, 7]
        detailed_trajectory = {interpret_level[i]: val for i, val in enumerate(tree_trajectory)}
        if detailed_trajectory["selected_parent_entity"] == 0:
            # Child entity
            entity.entity_names.add(self.child_entity_name + detailed_trajectory["selected_child_entity"])

            # Add parent name
            parent_name = self.parent_entity_names[detailed_trajectory["selected_parent_entity"]] + detailed_trajectory["selected_group"]
            entity.parent_names.add(parent_name)

            # Set up action type
            action_type = self.action_types[detailed_trajectory["selected_action_type"]]
            try:
                entity.parent_to_actions[parent_name].add(action_type)
            except KeyError:
                entity.parent_to_actions[parent_name] = {action_type}

            # Add to pattern
            pattern_symbol = self.all_pattern_symbols[action_type][detailed_trajectory["pattern_symbol"]]
            try:
                self.actions_to_patterns[action_type]
            except KeyError:
                self.actions_to_patterns

            self.parents_to_actions = set() if parents_to_actions is None else parents_to_actions
            self.actions_to_patterns = set() if actions_to_patterns is None else actions_to_patterns

        else:
            # Parent entity


        return entity
