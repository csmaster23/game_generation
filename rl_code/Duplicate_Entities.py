import math
import torch

class Duplicate_Entities_Model():
    def __init__(self, mechanic_types, embedding_size=240, layer_embedding_size=10, num_layers=24):
        self.mechanic_types = mechanic_types
        self.parent_min_max, self.child_min_max = None, None
        self.embedding_size = embedding_size

        self.d_model = layer_embedding_size * num_layers
        self.generator_action_network = torch.nn.Transformer(layer_embedding_size * num_layers,
                                                             num_encoder_layers=2,
                                                             num_decoder_layers=2,
                                                             dim_feedforward=100,
                                                             nhead=4)
        self.gen_embedder = torch.nn.Embedding(100, 5, padding_idx=0)
        self.softmax = torch.nn.Softmax(0)
        self.linear = torch.nn.Linear(layer_embedding_size * num_layers, embedding_size)


    def transformer_duplicate(self, embeddings, trajectories, comb_map, is_child=True):
        maxes = []
        min_max = self.get_all_min_max_dict(embeddings, trajectories, comb_map)
        # if is_child:
        #     self.child_min_max = self.get_child_min_max_dict(embeddings, trajectories, comb_map)
        #     min_max = self.child_min_max
        # else:
        #     self.parent_min_max = self.get_parent_min_max_dict(embeddings, trajectories, comb_map)
        #     min_max = self.parent_min_max
        # for i, row in enumerate(embeddings):
        for i in reversed(range(embeddings.shape[0])):
            row = embeddings[i]
            with torch.no_grad():
                src = row
            src = src.unsqueeze(0)
            src = src.unsqueeze(0)
            out = self.linear(self.generator_action_network(src, src[-1, None, :])).squeeze(0)
            mask = self.get_mask(min_max[i][0]) # this gets the min max tuple for this embedding
            out = out.squeeze(0) + mask
            probs = self.softmax(out)
            if torch.all(torch.isnan(probs)):
                chosen_num = 0
                # for i, row in enumerate(probs):
                #     if all(torch.isnan(row)):
                #         probs[i] = torch.zeros(row.shape)
            else:
                chosen_num = torch.argmax(probs).item()
            min_max = self.update_min_max(chosen_num, min_max, comb_map, embeddings, i)
            maxes.append(chosen_num)
        maxes.reverse() # switches it back to the correct embedding order because we reversed for the for loop ^^
        for key in min_max.keys():
            min_max[key] = [maxes[key], min_max[key][1]]
        return min_max  # this consists of a dict that has embedding num as key and the value is a list where there
                        # are two elements, the first element is the number of times it should be duplicated
                        # the second element is a list of all the original entities that make up the entity

    def update_min_max(self, chosen_num, min_max, comb_map, embeddings, i):
        # print("HERE boys!!!")
        min_max[i][0] = (min_max[i][0][0], min_max[i][0][1] - chosen_num) # this updates the current embedding min max we are looking at
        for emb_num in min_max[i][1]: # loops through the origins of embedding i
            if emb_num == i:
                continue
            min_max[emb_num][0] = (min_max[emb_num][0][0], min_max[emb_num][0][1] - chosen_num)
            # now we need to loop through the origins of the other embeddings to see if it also has the origins of emb_num
            for mm in min_max.keys():
                if mm == i:
                    continue
                if emb_num in min_max[mm][1]:
                    min_max[mm][0] = (min_max[mm][0][0], min_max[mm][0][1] - chosen_num)
        return min_max



    def get_all_min_max_dict(self, embeddings, trajectories, comb_map):
        min_max = {} # dictionary of embedding number as key and value that is list with two entries, first is tuple of min max and second is list of original entity combiners
        for i, emb in enumerate(embeddings):
            if i < len(trajectories):                                                   # these are embeddings we have trajectories for
                if len(trajectories[i].shape) == 1:
                    if trajectories[i][0] == self.mechanic_types['Square-Grid Movement']:
                        min_max[i] = [(5, 8), []] # the empty list denotes that nothing goes to combine an original entity
                    elif trajectories[i][0] == self.mechanic_types['Betting']:
                        min_max[i] = [(2, 4), []]
                    else:
                        min_max[i] = [(1, 1), []] # idk what this would be but it's here
                else:
                    if trajectories[i][0][0] == self.mechanic_types['Square-Grid Movement']:
                        min_max[i] = [(3, 6), []]  # the empty list denotes that nothing goes to combine an original entity
                    elif trajectories[i][0][0] == self.mechanic_types['Betting']:
                        min_max[i] = [(2, 4), []]
                    else:
                        min_max[i] = [(1, 1), []]  # idk what this would be but it's here
            else:                                                                       # embeddings that we just created through combinations that we have no trajectories for
                indices = self.get_key( emb, comb_map )                                 # will be in embedding order
                first_min_max = min_max[indices[0]][0][1]       # [0][1] to get tuple of min max and single out max
                second_min_max = min_max[indices[1]][0][1]      # [0][1] to get tuple of min max and single out max
                first_origins_lst = min_max[indices[0]][1]      # gets list of combined origins from indices
                second_origins_lst = min_max[indices[1]][1]     # gets list of combined origins from indices
                min_max[i] = [( 1, min([first_min_max, second_min_max]) ), first_origins_lst + second_origins_lst + [indices[0], indices[1]]]

        return min_max


    def get_mask(self, N ): # N in comes in as a tuple of min, max
        full_zeros = torch.zeros((self.embedding_size,))
        for i in range(full_zeros.shape[0]):
            if i >= N[1] or i < N[0]:
                full_zeros[i] = -math.inf
        return full_zeros

    # function to return key for any value
    def get_key(self, val, my_dict):
        for key, value in my_dict.items():
            if torch.all(torch.eq(val, value)):
                return key
        return "key doesn't exist"

    # def get_parent_min_max_dict(self, embeddings, trajectories, comb_map):
    #     min_max = {} # dictionary of embedding number as key and value that is list with two entries, first is tuple of min max and second is list of original entity combiners
    #     for i, emb in enumerate(embeddings):
    #         if i < len(trajectories):                                                   # these are embeddings we have trajectories for
    #             if trajectories[i][0] == self.mechanic_types['Square-Grid Movement']:
    #                 min_max[i] = [(5, 8), []] # the empty list denotes that nothing goes to combine an original entity
    #             elif trajectories[i][0] == self.mechanic_types['Betting']:
    #                 min_max[i] = [(2, 4), []]
    #             else:
    #                 min_max[i] = [(1, 1), []] # idk what this would be but it's here
    #         else:                                                                       # embeddings that we just created through combinations that we have no trajectories for
    #             indices = self.get_key( emb, comb_map )                                 # will be in embedding order
    #             first_min_max = min_max[indices[0]][0][1]       # [0][1] to get tuple of min max and single out max
    #             second_min_max = min_max[indices[1]][0][1]      # [0][1] to get tuple of min max and single out max
    #             first_origins_lst = min_max[indices[0]][1]      # gets list of combined origins from indices
    #             second_origins_lst = min_max[indices[1]][1]     # gets list of combined origins from indices
    #             min_max[i] = [( 1, min([first_min_max, second_min_max]) ), first_origins_lst + second_origins_lst + [indices[0], indices[1]]]
    #
    #     return min_max
    #
    # def get_child_min_max_dict(self, embeddings, trajectories, comb_map):
    #     min_max = {}  # dictionary of embedding number as key and value that is list with two entries, first is tuple of min max and second is list of original entity combiners
    #     for i, emb in enumerate(embeddings):
    #         if i < len(trajectories):  # these are embeddings we have trajectories for
    #             if trajectories[i][0][0] == self.mechanic_types['Square-Grid Movement']:
    #                 min_max[i] = [(3, 6), []]  # the empty list denotes that nothing goes to combine an original entity
    #             elif trajectories[i][0][0] == self.mechanic_types['Betting']:
    #                 min_max[i] = [(2, 4), []]
    #             else:
    #                 min_max[i] = [(1, 1), []]  # idk what this would be but it's here
    #         else:  # embeddings that we just created through combinations that we have no trajectories for
    #             indices = self.get_key(emb, comb_map)  # will be in embedding order
    #             first_min_max = min_max[indices[0]][0][1]  # [0][1] to get tuple of min max and single out max
    #             second_min_max = min_max[indices[1]][0][1]  # [0][1] to get tuple of min max and single out max
    #             first_origins_lst = min_max[indices[0]][1]  # gets list of combined origins from indices
    #             second_origins_lst = min_max[indices[1]][1]  # gets list of combined origins from indices
    #             min_max[i] = [(1, min([first_min_max, second_min_max])),
    #                           first_origins_lst + second_origins_lst + [indices[0], indices[1]]]
    #     return min_max

    # mechanic_types = {
    #     "Square-Grid Movement": 1,
    #     "Betting": 2,
    # }


    # indices = self.get_key(embeddings[i], comb_map)
    # if 'str' in str(type(indices)):       # original entity, i.e. non combined entity
    #     min_max[i] = (min_max[i][0], min_max[i][1] - chosen_num)# subtract the chosen num from the max of the corresponding min_max entry for embed row i
    # else:                                   # combined entity
    #     ind_one = indices[0]
    #     ind_two = indices[1]
    #     one_indices = self.get_key(embeddings[ind_one], comb_map)
    #     if 'tuple' in str(type(one_indices)): # means you've hit the base
    #         min_max[one_indices[0]] = (min_max[one_indices[0]][0], min_max[one_indices[0]][1] - chosen_num)
    #         min_max[one_indices[1]] = (min_max[one_indices[1]][0], min_max[one_indices[1]][1] - chosen_num)
    #
    #     two_indices = self.get_key(embeddings[ind_two], comb_map)
    #     if 'tuple' in str(type(two_indices)): # means you've hit the base
    #         min_max[two_indices[0]] = (min_max[two_indices[0]][0], min_max[two_indices[0]][1] - chosen_num)
    #         min_max[two_indices[1]] = (min_max[two_indices[1]][0], min_max[two_indices[1]][1] - chosen_num)
    #
    #     min_max[ind_one] = (min_max[ind_one][0], min_max[ind_one][1] - chosen_num)
    #     min_max[ind_two] = (min_max[ind_two][0], min_max[ind_two][1] - chosen_num)
    #     min_max[i] = (min_max[i][0], min_max[i][1] - chosen_num)
    #
    # return min_max

    # return min_max
    # min_max = {}
    # for i, traj in enumerate(trajectories):
    #     if traj[0][0] == self.mechanic_types['Square-Grid Movement']:
    #         min_max[i] = (3, 6)
    #     elif traj[0][0] == self.mechanic_types['Betting']:
    #         min_max[i] = (2, 4)
    #     else:
    #         min_max[i] = (1, 1)
    # return min_max
    # interpret_level = {
    #     0: "mechanic_num",
    #     1: "num_groups",
    #     2: "selected_group",
    #     3: "selected_parent_entity",  # Defaults to 0
    #     4: "num_child_entities",
    #     5: "selected_child_entity",
    #     6: "num_action_types",
    #     7: "selected_action_type",
    #     8: "num_patterns",
    #     9: "selected_pattern",
    #     10: "pattern_length",
    #     11: "pattern_symbol"
    # }