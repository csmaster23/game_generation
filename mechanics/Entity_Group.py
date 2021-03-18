import uuid
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

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
		self.id_to_idx = dict()
		self.idx_to_id = dict()
		self.id_to_parent_name = dict()
		self.adj_matrices = dict()
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
				self.idx_to_id[cur_idx] = entity_id
				self.id_to_parent_name[entity_id] = parent_name
				cur_idx += 1

	def create_adjacency_matrices(self, mechanic_obj):
		all_indices = np.arange(len(self.id_to_idx))
		for parent_name in self.parents_to_ids:
			cur_indices = []
			for entity_id in self.parents_to_ids[parent_name]:
				cur_indices.append(self.id_to_idx[entity_id])
			parent_adj_matrices = mechanic_obj.create_adjacency_matrices(all_indices, cur_indices, parent_name)
			self.adj_matrices[parent_name] = parent_adj_matrices

	def get_cmap(self, n, name='viridis'):
		'''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
		RGB color; the keyword argument name must be a standard mpl colormap name.'''
		return plt.cm.get_cmap(name, n)

	def visualize(self, ids_to_objects):
		for parent_type in self.parents_to_ids:
			if 'square' in parent_type:
				label_list = []
				square_ids = self.parents_to_ids[parent_type]
				for square_id in square_ids:
					entity_object = ids_to_objects[square_id]
					idx = self.id_to_idx[square_id]
					label = parent_type
					for name in entity_object.entity_names:
						if name != parent_type:
							label += " and "
							label += name
					label_list.append(label)

				unique_labels = np.unique(label_list)
				label_map = {label: i for i, label in enumerate(unique_labels)}
				cmap = self.get_cmap(len(unique_labels))
				l = int(np.sqrt(len(square_ids)))
				data_list = np.array([label_map[label] for label in label_list])
				data = data_list.reshape((l,l))

				# create discrete colormap
				fig, ax = plt.subplots()
				ax.imshow(data, cmap=cmap, label=label_list)

				ax.set_title(parent_type + " grid")
				# draw gridlines
				[ax.plot(0,0,c=cmap(label_map[label]), label=label_list[i])for i, label in enumerate(label_list)]

				ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
				ax.set_xticks(np.arange(-.5, l, 1))
				ax.set_yticks(np.arange(-.5, l, 1))
				ax.legend(loc="upper left")

			plt.show()




