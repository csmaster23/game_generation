import math
import torch
from torch import nn
import numpy as np
from functools import reduce
# from rl_code.Multi_Head_Attention import MultiheadAttention as MA

def get_factors(n):
    return set(reduce(list.__add__,([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

class Initializer_Model(nn.Module):
    def __init__(self, embedding_size=240):
        super().__init__()
        ## Make a 40 and 80 block for two rounds of combinations
        factor = max(get_factors(embedding_size)) # this is the number of heads to use, num_entities must be divisble by num_heads hence we get factors and pick largetst one
        self.m_att = torch.nn.MultiheadAttention(embedding_size, factor)
        self.softmax = torch.nn.Softmax(0)
        self.embedding_size = embedding_size
        self.symmetric_pattern_map = {"NW": "SE", "N": "S", "NE": "SW", "W": "E", "E": "W", "SW": "NE", "S": "N", "SE": "NW"}


    def find_initials(self, embeddings, verbose=False):#, mask):
        # TRAIN WITH THIS FUNCTION
        embeddings = embeddings.unsqueeze(0)
        if verbose:
            print("\n--- top of initialize attention ---")
            print("Embedding size: %s" % str(embeddings.size()))

        att_2, weights_2 = self.m_att(embeddings, embeddings, embeddings)
        att_2 = att_2.squeeze(0)
        dotted = torch.mm(att_2, att_2.T)
        summed = torch.sum(dotted, dim=1)

        # masked = dotted + mask

        # Soft-max the result so can compare to a threshold between 0 and 1
        soft_dot = self.softmax(summed)
        # for i, row in enumerate(soft_dot):
        if all(torch.isnan(soft_dot)):
            soft_dot = torch.zeros( soft_dot.shape )

        # Here we make the output matrix symmetric before we find the entities that meet the threshold to be combined
        # symmetric_out = (soft_dot.T + soft_dot) / 2
        return {"soft_dot": soft_dot, "logits": summed}

def initialize_some_entities(entity_dict, initializer_model, duplicated_embeddings, duplicate_combined_dict, verbose=False):
    att_thresh = 0.3
    if verbose:
        print("\n--- Top of Initialize Entity Positions in Game.py ---")
        print("Entity obj dict len: %s" % str(len(entity_dict.keys())))
    parent_keys, parent_embeds, child_keys, child_embeds = [], [], [], []
    all_keys = list(entity_dict.keys())
    for key in all_keys:
        entity = entity_dict[key]
        square_parent = False
        for ent_name in entity.entity_names:
            if 'square' in ent_name and 'square_movement' not in ent_name:
                parent_keys.append(key)
                parent_embeds.append(entity.get_embedding())
                square_parent = True
                break

        if not square_parent:
            child_keys.append(key)
            child_embeds.append(entity.get_embedding())

    # the 4 lists are now ordered and match up directly
    parent_tensors = torch.stack(parent_embeds)
    child_tensors = torch.stack(child_embeds)

    parent_out = initializer_model.find_initials(parent_tensors)

    # place the parents in correct order and give them a parent order
    sq_p_keys = []
    for p_key in parent_keys:
        highest_idx = torch.argmax(parent_out["soft_dot"]).item()
        entity_dict[p_key].parent_order = highest_idx
        entity_dict[p_key].storage_location = highest_idx
        parent_out["soft_dot"][highest_idx] = -math.inf
        for ent_name in entity_dict[p_key].entity_names:
            if 'square' in ent_name:
                sq_p_keys.append(p_key)
                break

    # handle child squares first
    sq_c_keys = []
    for c_key in child_keys:
        for ent_name in entity_dict[c_key].entity_names:
            if "square_movement" in ent_name:
                if verbose:
                    print(entity_dict[c_key].entity_names)
                sq_c_keys.append(c_key)
                break

    # split sq_c pieces evenly
    player_1, player_2, embeds = [], [], []
    i = 0
    while (i < len(sq_c_keys)):
        player_1.append(sq_c_keys[i])
        embeds.append( entity_dict[sq_c_keys[i]].get_embedding())
        player_2.append(sq_c_keys[i+1])
        i += 2

    # do attention on the player pieces
    stacked_embeds = torch.stack(embeds)
    child_sq_out = initializer_model.find_initials(stacked_embeds)

    # get the new order for the child pieces
    new_p1_keys, new_p2_keys = [], []
    for _ in player_1:
        highest_idx = torch.argmax(child_sq_out["soft_dot"]).item()
        child_sq_out["soft_dot"][highest_idx] = -math.inf
        new_p1_keys.append(player_1[highest_idx])
        new_p2_keys.append(player_2[highest_idx])

    # place the player 1 child entities in the first few parent entities
    for i, p1_k in enumerate(new_p1_keys):
        parent_id = None
        for key in all_keys:
            if entity_dict[key].parent_order == i: # meaning the parent we want
                entity_dict[key].my_stored_ids.append(p1_k)
                parent_id = key
                break
        entity_dict[p1_k].storage_location = parent_id
        entity_dict[p1_k].entity_names.add('player_1')

        # Make the pieces move symmetric
        for action_type in entity_dict[p1_k].actions_to_patterns:
            try:
                for pattern_num in entity_dict[p1_k].actions_to_patterns[action_type]:
                    new_pattern = []
                    for symbol in entity_dict[p1_k].actions_to_patterns[action_type][pattern_num]:
                        new_pattern.append(initializer_model.symmetric_pattern_map[symbol])
                    entity_dict[p1_k].actions_to_patterns[action_type][pattern_num] = new_pattern
            except KeyError:
                pass

        # Remove duplicate patterns

        new_actions_to_patterns = dict()
        for action_type in entity_dict[p1_k].actions_to_patterns:
            previous_patterns = set()
            new_actions_to_patterns[action_type] = dict()
            for pattern_num in entity_dict[p1_k].actions_to_patterns[action_type]:
                pattern = entity_dict[p1_k].actions_to_patterns[action_type][pattern_num]
                tuple_pattern = tuple(pattern)
                if tuple_pattern not in previous_patterns:
                    new_actions_to_patterns[action_type][pattern_num] = pattern
                    previous_patterns.add(tuple_pattern)

        entity_dict[p1_k].actions_to_patterns = new_actions_to_patterns



    # place the player 2 child entities in the last few parent entities
    for i, p2_k in enumerate(new_p2_keys):
        sq_parents_num = len(parent_keys) - 1
        parent_id = None
        for key in all_keys:
            if entity_dict[key].parent_order == (sq_parents_num-i):  # meaning the parent we want
                entity_dict[key].my_stored_ids.append(p2_k)
                parent_id = key
                break
        entity_dict[p2_k].storage_location = parent_id
        entity_dict[p2_k].entity_names.add('player_2')

        # Remove duplicate patterns
        new_actions_to_patterns = dict()
        for action_type in entity_dict[p2_k].actions_to_patterns:
            previous_patterns = set()
            new_actions_to_patterns[action_type] = dict()
            for pattern_num in entity_dict[p2_k].actions_to_patterns[action_type]:
                pattern = entity_dict[p2_k].actions_to_patterns[action_type][pattern_num]
                tuple_pattern = tuple(pattern)
                if tuple_pattern not in previous_patterns:
                    new_actions_to_patterns[action_type][pattern_num] = pattern
                    previous_patterns.add(tuple_pattern)

        entity_dict[p2_k].actions_to_patterns = new_actions_to_patterns

    # place the cards in the draw pile
    more_parents = len(parent_keys)
    all_unknown_children_keys = []
    draw_pile_key = None
    for a in all_keys:
        # skip entities that already have a storage location which will be set above for the squares
        if entity_dict[a].storage_location is None: # no location has been set for them
            if len(entity_dict[a].parent_names) == 0: # meaning a parent that has no location
                entity_dict[a].storage_location = more_parents
                more_parents += 1
            else: # meaning children who have no location
                all_unknown_children_keys.append(a)

        if 'draw_1' in entity_dict[a].entity_names:
            draw_pile_key = a

    # loop through unknown children and place them in a draw pile
    draw_pile = []
    for c_k in all_unknown_children_keys:
        draw_pile.append(c_k)
         # Makes sure any pieces on the are the last
        entity_dict[c_k].storage_location = draw_pile_key

    entity_dict[draw_pile_key].my_stored_ids = draw_pile + entity_dict[draw_pile_key].my_stored_ids

    for key in all_keys:
        entity = entity_dict[key]
        if verbose:
            print(str(entity))
        # Don't know why we have something that connects to all the entities?
        # if type(entity_dict[key]) is list:
        #     del entity_dict[key]

    return entity_dict


def create_masks(entity_dict, verbose=False):
    all_parent_names, all_masks = {}, {}
    for key in entity_dict.keys():
        parents = entity_dict[key].parent_names                 # grab parent names
        if len(parents) == 0:                                   # means that they are the parent entities
            if verbose:
                print("Names: %s" % str(entity_dict[key].entity_names))
            names = entity_dict[key].entity_names               # get names of parents
            for n in names:
                try:
                    all_parent_names[n] += 1
                except KeyError:
                    all_parent_names[n] = 1
    if verbose:
        print("All parent names: %s" % str(all_parent_names))
    for key in list(all_parent_names.keys()):
        if 'quare' in key:
            all_masks[key] = make_mask(all_parent_names[key], square=True)
        else:
            all_masks[key] = make_mask(all_parent_names[key], square=False)
    return all_masks


def make_mask(size, square=False):
    if square:
        squared = int(size**(1/2))
        return torch.zeros((squared,squared))
    else:
        return torch.zeros((size,))