import torch
from functools import reduce

def get_factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

class Attention_Model():
    def __init__(self, embedding_size=40, num_entities=4):
        factor = max(get_factors(num_entities)) # this is the number of heads to use, num_entities must be divisble by num_heads hence we get factors and pick largetst one
        self.multihead_attn = torch.nn.MultiheadAttention(num_entities, factor)
        self.softmax = torch.nn.Softmax(1)
        self.query = torch.rand((embedding_size, num_entities), dtype=torch.float32)
        self.key = torch.rand((embedding_size, num_entities), dtype=torch.float32)
        self.value = torch.rand((embedding_size, num_entities), dtype=torch.float32)


    def find_attention(self, embeddings):
        print("top of find attention")
        print("Embedding size: %s" % str(embeddings.size()))
        q = torch.matmul(embeddings, self.query).unsqueeze(0)
        k = torch.matmul(embeddings, self.key).unsqueeze(0)
        v = torch.matmul(embeddings, self.value).unsqueeze(0)
        attn_output, attn_output_weights = self.multihead_attn(q, k, v)
        return attn_output.squeeze(0), attn_output_weights.squeeze(-1)

def do_some_attention(embeddings, attention_model):
    print("top of do some attention")
    print("Embeddings with Tensor size: %s" % str(embeddings.size()))
    att_out, out_weights = attention_model.find_attention(embeddings)
    print("Size of att_out: %s\tsize of out_weights: %s" % (str(att_out.size()), str(out_weights.size())))
    return att_out#, out_weights