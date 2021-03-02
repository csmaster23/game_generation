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

class EntityGroup():
	def __init__(self, name):
		self.name = name
		self.parents_to_ids = dict()
	def add_entity_to_group(self, entity):
		for group_tuple in entity.entity_groups:
			parent_name = group_tuple[0]
			group_name = group_tuple[1]
			if group_name == self.name:
				try:
					self.parents_to_ids[parent_name].append(entity.id)
				except KeyError:
					self.parents_to_ids[parent_name] = [entity.id]
	def assign_entity_indices(self):
		cur_idx = 0
		for parent_name in self.parents_to_ids.keys():
			for entity_id in self.parents_to_ids[parent_name]:
				self.id_to_idx[entity_id] = cur_idx
				cur_idx += 1


