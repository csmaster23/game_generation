import math
import torch
import numpy as np
from functools import reduce
# from rl_code.Multi_Head_Attention import MultiheadAttention as MA

def get_factors(n):
    return set(reduce(list.__add__,([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

class Initializer_Model():
    def __init__(self, embedding_size=60, embedding_size_2=120):
        ## Make a 40 and 80 block for two rounds of combinations
        factor = max(get_factors(embedding_size)) # this is the number of heads to use, num_entities must be divisble by num_heads hence we get factors and pick largetst one
        factor_2 = max(get_factors(embedding_size_2))
        self.m_att = torch.nn.MultiheadAttention(embedding_size, factor)
        self.m_att_2 = torch.nn.MultiheadAttention(embedding_size_2, factor_2)
        self.softmax = torch.nn.Softmax(1)
        self.embedding_size = embedding_size
        self.embedding_size_2 = embedding_size_2


    def find_initials(self, embeddings, mask):
        embeddings = embeddings.unsqueeze(0)
        print("\n--- top of find attention ---")
        print("Embedding size: %s" % str(embeddings.size()))
        if embeddings.shape[-1] == self.embedding_size:
            att_2, weights_2 = self.m_att(embeddings, embeddings, embeddings)
        else:
            att_2, weights_2 = self.m_att_2(embeddings, embeddings, embeddings)

        att_2 = att_2.squeeze(0)
        dotted = torch.mm(att_2, att_2.T)

        # Here we mask out entities so they can't be combined with entities from their same mechanic
        masked = dotted + mask

        # Soft-max the result so can compare to a threshold between 0 and 1
        soft_dot = self.softmax(masked)
        for i, row in enumerate(soft_dot):
            if all(torch.isnan(row)):
                soft_dot[i] = torch.zeros( row.shape )

        # Here we make the output matrix symmetric before we find the entities that meet the threshold to be combined
        symmetric_out = (soft_dot.T + soft_dot) / 2
        return symmetric_out

def initialize_some_entities(entity_dict, initializer_model, entity_groups, duplicated_embeddings, duplicate_combined_dict):
    att_thresh = 0.3
    print("\n--- Top of Initialize Entity Positions in Game.py ---")
    print("Entity obj dict len: %s" % str(len(entity_dict.keys())))
    p_count, c_count = 0,0
    for key in entity_dict.keys():
        print("\nKey: %s" % str(key))
        print("Value: %s" % str(entity_dict[key].__dict__))
        if len(entity_dict[key].parent_names) == 0:
            p_count += 1
        else:
            c_count += 1
    counter_ = 0
    for i, key in enumerate(duplicate_combined_dict):
        dup_num = duplicate_combined_dict[key][0]  # gets the duplication number for this embedding
        for d in range(dup_num):

            counter_ += 1



    return entity_dict


def create_masks(entity_dict):
    all_parent_names, all_masks = {}, {}
    for key in entity_dict.keys():
        parents = entity_dict[key].parent_names                 # grab parent names
        if len(parents) == 0:                                   # means that they are the parent entities
            print("Names: %s" % str(entity_dict[key].entity_names))
            names = entity_dict[key].entity_names               # get names of parents
            for n in names:
                try:
                    all_parent_names[n] += 1
                except KeyError:
                    all_parent_names[n] = 1
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