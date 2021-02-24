import uuid

interpret_level = {
      0: "mechanic_num",
      1: "num_groups",
      2: "selected_group",
      3: "selected_parent_entity", # Defaults to 0
      4: "num_child_entities",
      5: "selected_child_entity",
      6: "num_action_types",
      7: "selected_action_type",
      8: "num_patterns",
      9: "selected_pattern",
      10: "pattern_length",
      11: "pattern_symbol"
    }

class Entity():
    def __init__(self, entity_names=None, parent_names=None, parents_to_actions=None, actions_to_patterns=None, id=None, p=None):
        """
        :param entity_names: List of names for of each piece type that the entity belongs to. Each name includes a generic
        name (from the mechanic) and a subtype.
        :param parent_names: Names of the parents of this entity. This specifies which entity types it can move across.
        Each name includes a generic name (from the mechanic) and a subtype.
        :param parents_to_actions: A dictionary mapping a parent name to a list of action types (strings from mechanic)
        :param actions_to_patterns: A dictionary mapping an action type (string) to an action pattern (a sequence of entity
        group actions)
        :param id: An id that can be used to identify this distinct entity in the future
        :param p: A dictionary of parameters to fill in any more information
        """
        self.p = p
        self.entity_names = set() if entity_names is None else entity_names
        self.parent_names = set() if parent_names is None else parent_names
        self.parents_to_actions = dict() if parents_to_actions is None else parents_to_actions
        self.actions_to_patterns = dict() if actions_to_patterns is None else actions_to_patterns
        if id is None:
            self.id = uuid.uuid1()
        else:
            self.id = id
        self.storage_location = None
        self.my_stored_ids = []
