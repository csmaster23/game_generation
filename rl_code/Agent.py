from mechanics.Entity import Entity
import numpy as np
import torch

class RandomAgent():
    def __init__(self):
        pass

    def take_action(self, trajectories, min_choice, max_choice):
        return np.random.randint(min_choice, max_choice+1)

class CreatorAgent():
    def __init__(self, layer_embedding_size=4, num_layers=9):
        self.d_model = layer_embedding_size*num_layers
        self.generator_action_network = torch.nn.Transformer(layer_embedding_size*num_layers,
                                                             num_encoder_layers=2,
                                                             num_decoder_layers=2,
                                                             dim_feedforward=100,
                                                             nhead=4)
        self.gen_embedder = torch.nn.Embedding(100, 4, padding_idx=0)
        self.softmax = torch.nn.Softmax(1)


    def take_action(self, trajectories, constraints):

        with torch.no_grad():
            N = len(trajectories)
            src = self.gen_embedder(torch.cat(trajectories)).reshape(N, 1, -1)

        mask = self.generator_action_network.generate_square_subsequent_mask(N)
        out = self.generator_action_network(src, src[-1,None,:]) # May need to fix this later. Not sure what the tgt output should be
        probs = self.softmax(out.view(-1,self.d_model))

        if type(constraints) == torch.tensor:
            choices = torch.round(torch.sum(probs * constraints,dim=1))
        else:
            scale = torch.from_numpy(np.linspace(constraints[0],constraints[1],probs.shape[-1]))
            choices = torch.round(torch.sum(probs * scale,dim=1))

        return choices, probs

class Agent():
    def __init__(self, p):
        self.p = p


        # "Square-Grid Movement"
        # "Static Capture"

    def generate_entities(self):
        '''
        params:
        :return: list of generic entities
        '''
        print("\n--- Top of Generate Entities ---")
        print("Mechanic List: %s" % str(self.p['mechanics']))
        sq = "Square-Grid Movement"
        if sq in self.p['mechanics']:
            h_w_tuple = self.choose_grid_size()
            grids = self.create_grids(h_w_tuple, sq)
            for grid in grids:
                print("Grid Id: %s" % grid.id)
                num_pieces = self.choose_num_pieces()
        print("Done...")
        return []

    def choose_grid_size(self):
        return (5,5)
    def choose_num_pieces(self):
        return 1
    def create_grids(self, h_w_tuple, type_):
        grids = []
        for h in range(h_w_tuple[0]):
            for w in range(h_w_tuple[1]):
                id_ = str(h) + " " + str(w)
                grids.append( Entity( type_, id_, self.p ) )
        return grids