import math
import torch
from torch import nn

class Duplicate_Entities_Model(nn.Module):
    def __init__(self, mechanic_types, mechanic_dicts, embedding_size=240, layer_embedding_size=10, num_layers=24, verbose=False):
        super().__init__()
        self.mechanic_types = mechanic_types
        self.mechanic_dicts = mechanic_dicts
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
        self.verbose=verbose


    def transformer_duplicate(self, embeddings, trajectories, comb_map, is_child=True):
        # TRAIN WITH THIS FUNCTION
        maxes = []
        chosed = []
        min_max = self.NEW_get_all_min_max_dict(embeddings, trajectories, comb_map)
        first = True
        for i in range(len(trajectories)):
            row = embeddings[i]
            with torch.no_grad():
                src = row
            src = src.unsqueeze(0)
            src = src.unsqueeze(0)
            out = self.linear(self.generator_action_network(src, src[-1, None, :])).squeeze(0)
            mask = self.get_mask(min_max[i][0])  # this gets the min max tuple for this embedding
            out = out.squeeze(0) + mask
            probs = self.softmax(out)
            if torch.all(torch.isnan(probs)):
                chosen_num = 0
            else:
                chosen_num = torch.argmax(probs).item()
            ## SQUARE IT BRUH
            if len(trajectories[i].shape) == 1:  # parent
                if trajectories[i][0] == self.mechanic_types['Square-Grid Movement'] and first: # first means square piece
                    chosen_num = chosen_num ** 2
                    first = False
                elif trajectories[i][0] == self.mechanic_types['Square-Grid Movement'] and not first:
                    chosen_num = 1
            ## SQUARE IT BRUH
            chosed.append(chosen_num)
        if self.verbose:
            print("Chosed: %s" % str(chosed))

        min_max = self.get_chose_dict( chosed, embeddings, comb_map )
        for i in reversed(range(embeddings.shape[0])): # go through combined now
            if i < (len(chosed)):
                break
            row = embeddings[i]
            with torch.no_grad():
                src = row
            src = src.unsqueeze(0)
            src = src.unsqueeze(0)
            out = self.linear(self.generator_action_network(src, src[-1, None, :])).squeeze(0)
            mask = self.get_mask(min_max[i][0])  # this gets the min max tuple for this embedding
            out = out.squeeze(0) + mask
            probs = self.softmax(out)
            if torch.all(torch.isnan(probs)):
                chosen_num = 0
            else:
                chosen_num = torch.argmax(probs).item()
            maxes.append(chosen_num)
            min_max = self.update_min_max(chosen_num, min_max, comb_map, embeddings, i)
        maxes.reverse()
        for key in min_max.keys():
            if key < len(trajectories): # originals
                min_max[key] = [min_max[key][0][1], min_max[key][1]]
            else:
                min_max[key] = [maxes[key-len(trajectories)], min_max[key][1]]
        return min_max

    def get_chose_dict( self, chosed, embeddings, comb_map ):
        new_map = {}
        for i, emb in enumerate(embeddings):
            if i < len(chosed): # means orginal
                new_map[i] = [ (chosed[i], chosed[i]), [] ]
            else:
                key = self.get_key(emb, comb_map)
                first_max = chosed[ key[0] ]
                second_max = chosed[ key[1]]
                lowest = min(first_max, second_max)
                first_origins_lst = new_map[key[0]][1]  # gets list of combined origins from indices
                second_origins_lst = new_map[key[1]][1]  # gets list of combined origins from indices
                # new_map[i] = [ (1, lowest), first_origins_lst + second_origins_lst + [key[0], key[1]]]
                new_map[i] = [(lowest, lowest), first_origins_lst + second_origins_lst + [key[0], key[1]]]
        return new_map

    def update_min_max(self, chosen_num, min_max, comb_map, embeddings, i):
        # print("HERE boys!!!")
        min_max[i][0] = (min_max[i][0][0], min_max[i][0][1] - chosen_num) # this updates the current embedding min max we are looking at
        for emb_num in min_max[i][1]: # loops through the origins of embedding i
            if emb_num == i:
                continue
            min_max[emb_num][0] = (min_max[emb_num][0][1] - chosen_num, min_max[emb_num][0][1] - chosen_num)
            # now we need to loop through the origins of the other embeddings to see if it also has the origins of emb_num
            for mm in min_max.keys():
                if mm == i:
                    continue
                if emb_num in min_max[mm][1]:
                    if min_max[emb_num][0][1] < min_max[mm][0][1]: # this means that we are looking at the correct max and not the bigger one
                        min_max[mm][0] = (1, min_max[mm][0][1] - chosen_num)
        return min_max

    def NEW_get_all_min_max_dict(self, embeddings, trajectories, comb_map):
        min_max = {} # dictionary of embedding number as key and value that is list with two entries, first is tuple of min max and second is list of original entity combiners
        for i, emb in enumerate(embeddings):
            if i < len(trajectories):                                                   # these are embeddings we have trajectories for
                if len(trajectories[i].shape) == 1:
                    # trajectories[i][0[ returns 1 or 2 depending on which mechanic it is
                    mechanic_num = trajectories[i][0].item()
                    min_max[i] = [self.mechanic_dicts[mechanic_num]["parent_dup"], []]
                    # if trajectories[i][0] == self.mechanic_types['Square-Grid Movement']:
                    #     min_max[i] = [(5, 8), []] # the empty list denotes that nothing goes to combine an original entity
                    # elif trajectories[i][0] == self.mechanic_types['Betting']:
                    #     min_max[i] = [(2, 4), []]
                    # else:
                    #     min_max[i] = [(1, 1), []] # idk what this would be but it's here
                else:
                    mechanic_num = trajectories[i][0][0].item()
                    min_max[i] = [self.mechanic_dicts[mechanic_num]["child_dup"], []]
                    # if trajectories[i][0][0] == self.mechanic_types['Square-Grid Movement']:
                    #     min_max[i] = [(3, 6), []]  # the empty list denotes that nothing goes to combine an original entity
                    # elif trajectories[i][0][0] == self.mechanic_types['Betting']:
                    #     min_max[i] = [(2, 4), []]
                    # else:
                    #     min_max[i] = [(1, 1), []]  # idk what this would be but it's here
            else:                                                                       # embeddings that we just created through combinations that we have no trajectories for
                pass
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
        if N[0] == N[1]: # meaning min is equal to max meaning we constrain model to only choose one num
            for i in range(full_zeros.shape[0]):
                if i == N[0]:
                    pass
                else:
                    full_zeros[i] = -math.inf
        else:
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