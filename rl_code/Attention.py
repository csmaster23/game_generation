import math
import torch
import numpy as np
from functools import reduce
from rl_code.Multi_Head_Attention import MultiheadAttention as MA

def get_factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

class Attention_Model():
    def __init__(self, embedding_size=40, embedding_size_2=80):
        ## Make a 40 and 80 block for two rounds of combinations
        factor = max(get_factors(embedding_size)) # this is the number of heads to use, num_entities must be divisble by num_heads hence we get factors and pick largetst one
        factor_2 = max(get_factors(embedding_size_2))
        self.m_att = MA(embedding_size, factor) # custom MultiHeadAttention (as code as pytorch just copied into py file)
        self.m_att_2 = MA(embedding_size_2, factor_2)
        self.softmax = torch.nn.Softmax(1)
        self.embedding_size = embedding_size
        self.embedding_size_2 = embedding_size_2


    def find_attention(self, embeddings, mask):
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

def do_some_attention(embeddings, trajectories, attention_model):
    att_thresh = 0.3
    print("top of do some attention")
    print("Embeddings with Tensor size: %s" % str(embeddings.size()))

    # index 2 of trajectories shows the group number and anything with the same group number will be masked out
    entity_group_nums = [traj[0][2].item() for traj in trajectories]

    mask = create_mask(entity_group_nums)

    att_out = attention_model.find_attention(embeddings, mask)

    indices_to_combine = find_combinations(att_out, threshold=att_thresh)
    indices_to_combine = remove_duplicates(indices_to_combine)
    mask = update_mask(mask, indices_to_combine, entity_group_nums)

    new_embeddings = []
    for embedding in embeddings:
        concatted = torch.cat( (embedding, embedding) )
        new_embeddings.append(concatted)
    for ind in indices_to_combine:
        concatted = torch.cat( (embeddings[ind[0]], embeddings[ind[1]]) )
        new_embeddings.append(concatted)
    all_embeddings = torch.stack(new_embeddings)

    # round 2 of attention
    all_att = attention_model.find_attention(all_embeddings, mask)
    indices_to_combine = indices_to_combine + find_combinations(all_att, threshold=att_thresh)
    indices_to_combine = remove_duplicates(indices_to_combine)
    print("Indices to combine second round: %s" % str(indices_to_combine))
    return indices_to_combine, all_embeddings


def update_mask(mask, indices, group_nums):
    rows, cols = [], []
    for i in indices:                                   # get the mask for the the new combination values row wise
        rows.append( ( mask[i[0]]+mask[i[1]] ) )
    for jk in indices:                                  # get the mask for each existing entity and the new combinations
        print(jk)
        col = []
        for n in group_nums:
            if group_nums[jk[0]] == n or group_nums[jk[1]] == n:
                col.append(-math.inf)
            else:
                col.append(0)
        cols.append(torch.FloatTensor(col))

    mask = torch.cat( (mask, torch.stack(cols).reshape(mask.shape[0],len(cols))), dim=1 ) # add col values to mask

    # row_addition = torch.FloatTensor( [-math.inf for _ in range(len(cols))] ) # this gets mask for relationship between combos
    row_additions = []
    for i, ind in enumerate(indices):
        row_add = []
        for j, ind_ in enumerate(indices):
            if i == j:
                row_add.append( -math.inf )
            elif group_nums[ind[0]] == group_nums[ind_[0]] or group_nums[ind[1]] == group_nums[ind_[0]] or \
                group_nums[ind[0]] == group_nums[ind_[1]] or group_nums[ind[1]] == group_nums[ind_[1]]:
                row_add.append( -math.inf )
            else:
                row_add.append(0)
        row_additions.append( torch.FloatTensor(row_add) )



    for rr in range(len(rows)):                             # this appends on that mask ^^ to the existing row masks
        rows[rr] = torch.cat( (rows[rr], row_additions[rr]) )
    mask = torch.cat((mask, torch.stack(rows)), dim=0)      # concat the rows to bottom of partial mask to get full mask

    print("Mask size: %s" % str(mask.size()))
    return mask

def create_mask(entity_group_nums):
    print("Entity Group Nums: %s" % str(entity_group_nums))
    mask = []
    for i, num in enumerate(entity_group_nums):
        inner_mask = []
        for j, num_ in enumerate(entity_group_nums):
            if i == j: # this being the entity paired with itself
                inner_mask.append( -math.inf ) # negative infinity is used to ensure the softmax puts this value at 0
            elif num == num_: # this being if the entities are of the same group type
                inner_mask.append(-math.inf)
            else:
                inner_mask.append(0) # zero because when we add the mask over we don't want the attention value to change
        mask.append(inner_mask)
    for i in range(len(mask)):
        mask[i] = torch.FloatTensor(mask[i])
    mask = torch.stack(mask)
    print("FINAL MASK: %s" % str(mask))
    return mask

# remove duplicate indices from tuple list, removes both kinds of duplices for (2,5) duplicated as (2,5) and (5,2)
def remove_duplicates(indices):
    sorted_list = [tuple(sorted(tup)) for tup in indices]
    new_list = list(set([i for i in sorted_list]))
    return new_list


def find_combinations(attention, threshold=.4):
    indices = []
    for i, row in enumerate(attention):
        for j, ele in enumerate(row):
            if ele >= threshold:
                if i == j:
                    pass
                else:
                    indices.append( (i, j) )
    return indices