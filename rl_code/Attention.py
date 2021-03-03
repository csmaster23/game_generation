import math
import torch
import numpy as np
from functools import reduce
# from rl_code.Multi_Head_Attention import MultiheadAttention as MA

def get_factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

class Attention_Model():
    def __init__(self, embedding_size=60, embedding_size_2=120):
        ## Make a 40 and 80 block for two rounds of combinations
        factor = max(get_factors(embedding_size)) # this is the number of heads to use, num_entities must be divisble by num_heads hence we get factors and pick largetst one
        factor_2 = max(get_factors(embedding_size_2))
        self.m_att = torch.nn.MultiheadAttention(embedding_size, factor) # custom MultiHeadAttention (as code as pytorch just copied into py file)
        self.m_att_2 = torch.nn.MultiheadAttention(embedding_size_2, factor_2) # WAS MA OLD
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

def do_some_attention(embeddings, trajectories, attention_model, is_child=False):
    att_thresh = 0.3
    print("top of do some attention")
    print("Embeddings with Tensor size: %s" % str(embeddings.size()))
    comb_to_emb_map = {}
    entity_group_nums, entity_mechanic_nums = [], []
    for traj in trajectories:
        if len(traj.shape) == 1:
            entity_group_nums.append( traj[2].item() )
            entity_mechanic_nums.append( traj[0].item() )
        else:
            entity_group_nums.append(traj[0][2].item())
            entity_mechanic_nums.append(traj[0][0].item())

    # if is_child:
    #     entity_group_nums = [traj[0][2].item() for traj in trajectories]
    #     entity_mechanic_nums = [traj[0][0].item() for traj in trajectories]
    # else:
    #     entity_group_nums = [traj[2] for traj in trajectories]
    #     entity_mechanic_nums = [traj[0] for traj in trajectories]
    custom_entity_groups = get_custom_entity_groups(entity_group_nums, entity_mechanic_nums)

    mask = create_mask(custom_entity_groups)

    att_out = attention_model.find_attention(embeddings, mask)

    first_indices_to_combine = find_combinations(att_out, threshold=att_thresh)
    first_indices_to_combine = remove_duplicates(first_indices_to_combine)
    print("Indices to combine first round: %s" % str(first_indices_to_combine))
    mask = update_mask(mask, first_indices_to_combine, custom_entity_groups)

    new_embeddings, old_doubles = [], []
    for embedding in embeddings:
        concatted = torch.cat( (embedding, embedding) )
        new_embeddings.append(concatted)
        old_doubles.append(concatted)
    for ind in first_indices_to_combine:
        concatted = torch.cat( (embeddings[ind[0]], embeddings[ind[1]]) )
        new_embeddings.append(concatted)
    all_embeddings = torch.stack(new_embeddings)

    # round 2 of attention
    all_att = attention_model.find_attention(all_embeddings, mask)
    second_indices_to_combine = first_indices_to_combine + find_combinations(all_att, threshold=att_thresh)
    second_indices_to_combine = remove_duplicates(second_indices_to_combine)
    print("Indices to combine second round: %s" % str(second_indices_to_combine))

    new_embeddings = []
    for embedding in old_doubles:           # we go through doubles so we can get the originals doubled again to their final size
        concatted = torch.cat((embedding, embedding))
        new_embeddings.append(concatted)

    for ind in first_indices_to_combine: #loop through first round of combinations (have to do this before we can do second so we can get correct embeddings concatted
        concatted = torch.cat((embeddings[ind[0]], embeddings[ind[1]]))     # looks at original sized embeddings to combine them correctly for first round combinations
        double_concatted = torch.cat((concatted, concatted))                # do this to double the first round of combined objects so we can have the correct final size for the entity embeddings for the first round of combinations
        comb_to_emb_map[ind] = double_concatted
        new_embeddings.append(double_concatted)

    # temp_embeddings = torch.stack(new_embeddings)
    for ind in second_indices_to_combine:
        if ind in first_indices_to_combine:
            continue
        concatted = torch.cat((all_embeddings[ind[0]], all_embeddings[ind[1]]))
        comb_to_emb_map[ind] = concatted
        new_embeddings.append(concatted)


    final_embeddings = torch.stack(new_embeddings)
    return second_indices_to_combine, final_embeddings, comb_to_emb_map


def update_mask(mask, indices, group_nums):
    if len(indices) == 0:
        return mask
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

def get_custom_entity_groups(entity_group_nums, entity_mechanic_nums):
    print("Entity group nums: %s" % str(entity_group_nums))
    print("Entity Mechanic nums: %s" % str(entity_mechanic_nums))
    custom_nums = []
    existing_tuples = []
    for i, (mech, group) in enumerate(zip(entity_mechanic_nums, entity_group_nums)):
        tup = (mech, group)
        if tup in existing_tuples:
            # custom_nums.append(len(existing_tuples)) # this will give us a custom group number for each unique mechanic group number pair
            custom_nums.append(index_that_matches(existing_tuples, tup))
        else:
            existing_tuples.append(tup)
            custom_nums.append(len(existing_tuples))

    return custom_nums

def index_that_matches(tup_lst, tup):
    for i, tup_l in enumerate(tup_lst):
        if tup == tup_l:
            return i + 1
    return 1
