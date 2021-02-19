import torch
from functools import reduce
# from rl_code.Multi_Head_Attention import MultiheadAttention as MA

def get_factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

class Attention_Model():
    def __init__(self, embedding_size=40, embedding_size_2=80):
        ## Make a 40 and 80 block for two rounds of combinations
        factor = max(get_factors(embedding_size)) # this is the number of heads to use, num_entities must be divisble by num_heads hence we get factors and pick largetst one
        factor_2 = max(get_factors(embedding_size_2))
        self.m_att = torch.nn.MultiheadAttention(embedding_size, factor)
        self.m_att_2 = torch.nn.MultiheadAttention(embedding_size_2, factor_2)

        # self.m_att = MA(embedding_size, factor) # custom MultiHeadAttention (as code as pytorch just copied into py file)
        # self.m_att_2 = MA(embedding_size_2, factor_2)
        self.softmax = torch.nn.Softmax(1)
        self.embedding_size = embedding_size
        self.embedding_size_2 = embedding_size_2


    def find_attention(self, embeddings):
        embeddings = embeddings.unsqueeze(0)
        print("\n--- top of find attention ---")
        print("Embedding size: %s" % str(embeddings.size()))
        if embeddings.shape[-1] == self.embedding_size:
            att_2, weights_2 = self.m_att(embeddings, embeddings, embeddings)
        else:
            att_2, weights_2 = self.m_att_2(embeddings, embeddings, embeddings)
        att_2 = att_2.squeeze(0)
        dotted = torch.mm(att_2, att_2.T)
        soft_dot = self.softmax(dotted)
        print("dotted shape: %s" % str(soft_dot.shape))

        return soft_dot

def do_some_attention(embeddings, attention_model):
    print("top of do some attention")
    print("Embeddings with Tensor size: %s" % str(embeddings.size()))
    att_out = attention_model.find_attention(embeddings)
    print("Size of att_out: %s" % str(att_out.size()))
    indices_to_combine = find_combinations(att_out, threshold=.2)
    return indices_to_combine

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