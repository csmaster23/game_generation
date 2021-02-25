import numpy as np
from mechanics.Entity import Entity
from mechanics.Mechanic import Mechanic
from rl_code.Game import interpret_level

class Deck_of_Cards_Class(Mechanic):
    def __init__(self, p=None):
        super().__init__(p)
        if p is None:
            self.p = {"num_cards" : 52,
                      "parent_entity_names": ["draw", "discard", "playing_area", "hand"],
                      "default_actions": ["move"],
                      "reflect_patterns": False,
                      "draw_discard": True,
                      "player_hand": False}
        self.mechanic_type = "Deck-of-Cards"
        self.num_cards = self.p['num_cards']
        self.parent_entity_names = {i+1: name for i, name in enumerate(self.p["parent_entity_names"])}
        self.child_entity_name = "card"
        self.action_types = {
                            "default": self.p['default_actions'],
                             1: "match",
                             2: "reverse_turn",
                             3: "increment_draw"
                             }

        # These pattern symbols can be used for all the actions
        self.optional_pattern_symbols = {
            1: "black", 2: "red", 3: "blue", 4: "green", 5: "1", 6: "2", 7: "3", 8: "4", 9: "5", 10: "6", 11: "7",
            12: "8", 13: "9", 14: "10", 15: "11", 16: "12", 17: "13"
        }
        self.default_pattern_symbols = {
            1: "transfer"
        }
        self.all_pattern_symbols = {
                                    "move" : self.pattern_symbols2,
                                     "match": self.pattern_symbols1,
                                     "reverse_turn": self.pattern_symbols1,
                                     "increment_draw": self.pattern_symbols1
                                    }

    def get_mechanic_dict(self):
        card = {}
        # Key[level]            (Min, Max)
        card["num_groups"] = (1, max([x for x in self.parent_entity_names.keys() if
                                    type(x) is int]))  # num_group (how many of that mechanic there is)
        # Should record which group we are on
        card["num_child_entities"] = (1, 6)  # num child entity types
        # Should record child entity type we are on
        card["num_action_types"] = (1, max([x for x in self.action_types.keys() if type(x) is int]))  # num_action_types
        card["num_patterns"] = (1, 4)  # num_patterns
        # Should record which pattern we are looking at
        card["pattern_length"] = (1, 3)  # pattern_length
        card["pattern_symbol"] = (
        1, max([x for x in self.optional_pattern_symbols.keys() if type(x) is int]))  # pattern_symbol
        card["num_parent_entity_types"] = len(self.parent_entity_names)
        return card

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