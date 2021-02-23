import math
import torch

class Duplicate_Entities_Model():
    def __init__(self, embedding_size=120, layer_embedding_size=10, num_layers=12):
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


    def transformer_duplicate(self, embeddings, trajectories, is_child=True):
        maxes = []
        for i, row in enumerate(embeddings):
            with torch.no_grad():
                if is_child:
                    N = trajectories[i][0][4] # this accesses number of child entities are max for this entity
                else:
                    N = trajectories[0][0][1] # this is num groups for parents
                # src = self.gen_embedder(row)
                src = row
            src = src.unsqueeze(0)
            src = src.unsqueeze(0)
            gen_mask = self.generator_action_network.generate_square_subsequent_mask(N)
            out = self.linear(self.generator_action_network(src, src[-1, None, :])).squeeze(0)
            mask = self.get_mask(N)
            out = out.squeeze(0) + mask
            probs = self.softmax(out)
            maxes.append(torch.argmax(probs).item())

        return maxes

    def get_mask(self, N ):
        full_zeros = torch.zeros((self.embedding_size,))
        for i in range(full_zeros.shape[0]):
            if i >= N:
                full_zeros[i] = -math.inf
        return full_zeros