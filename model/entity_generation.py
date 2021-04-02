from mechanics.Entity import Entity
import numpy as np
import torch
from torch import nn

from model.transformer_decoder import TransformerDecoderLayer, TransformerDecoder, PositionalEncoding


class EntityGenerationModel(nn.Module):
    def __init__(self, layer_embedding_size=5, num_layers=12):
        super().__init__()
        self.d_model = layer_embedding_size*num_layers
        self.generator_action_network = torch.nn.Transformer(layer_embedding_size*num_layers,
                                                             num_encoder_layers=2,
                                                             num_decoder_layers=2,
                                                             dim_feedforward=100,
                                                             nhead=4)
        self.gen_embedder = torch.nn.Embedding(100, 5, padding_idx=0)
        self.softmax = torch.nn.Softmax(1)
        self.linear = torch.nn.Linear(layer_embedding_size*num_layers,10)


    def get_decision(self, trajectories, mask=None):
        # TRAIN WITH THIS FUNCTION

        # May need to keep the embeddings here fixed
        N = len(trajectories)
        src = self.gen_embedder(torch.cat(trajectories)).reshape(N, 1, -1)

        out = self.linear(self.generator_action_network(src, src[-1,None,:])) # May need to fix this later. Not sure what the tgt output should be
        probs = self.softmax(out.view(-1) + mask.unsqueeze(0))

        # if type(constraints) == torch.tensor:
        #     choices = torch.round(torch.sum(probs * constraints,dim=1))
        # else:
        #     scale = torch.from_numpy(np.linspace(constraints[0],constraints[1],probs.shape[-1])).float()
        #     choices = torch.round(torch.sum(probs * scale,dim=1))
        return {"logits":out, "masked_probs":probs}


class AttentionLayer(nn.Module):
    """Attention layer implemented as in self-attention, but with a trainable prototype query
    vector instead of a query that is a transformation of the input. Justification: for the
    purposes of autoencoding and predicting, the prototype vector for summarizing the sequence
    does not depend on different tokens - it is always has the same job: summarize the sequence
    for e.g. an autoencoding task (as opposed to the job of predicting the next character, in
    which the prototype vector changes based on the character preceding the character to predict.)
    """

    def __init__(self, d_model):
        super().__init__()

        # TODO: add multiple heads
        # The query should be a trainable prototype vector of weights, such that multiplying Q by K^T is
        # just multiplying K by a linear layer from d_model to 1
        self.lin_q = nn.Linear(d_model, 1)
        self.lin_k = nn.Linear(d_model, d_model)
        self.lin_v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=2)
        self.scale_factor = torch.sqrt(torch.tensor(d_model, dtype=torch.float))

    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x ((N x L x d_model) torch.Tensor): the input embeddings
            return_attention_weights (bool): whether to return the tensor weights
                with the network output
        """
        # x: N x L x d_model
        k = self.lin_k(x)
        # k: N x L x d_model
        # This is where we differ from self-attention: we use a learnable prototype Q vector,
        # implemented as a linear layer, instead of transforming the input to get queries
        attn = self.lin_q(k)
        # attn: N x L x 1
        attn = torch.transpose(attn, 1, 2)
        # attn: N x 1 x L
        attn = attn / self.scale_factor
        attn = self.softmax(attn)
        # attn: N x 1 x L
        v = self.lin_v(x)
        # v: N x L x d_model
        out = torch.bmm(attn, v).squeeze(1)
        # out: N x d_model

        if return_attention_weights:
            return out, attn.squeeze(1)

        return out


class ManyToOneAttentionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.attn = AttentionLayer(d_model)
        self.lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(.1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x ((N x L x d_model) torch.Tensor): the input embeddings
            return_attention_weights (bool): whether to return the tensor weights
                with the network output
        """
        if return_attention_weights:
            # x: N x L x d_model
            out, attn_weights = self.attn(x, return_attention_weights=return_attention_weights)
            # out: N x 1 x d_model
            out = self.dropout(out)
            # out: N x 1 x d_model
            # out = self.layer_norm(out)
            # Experimenting with skip connection
            out = self.layer_norm(torch.mean(x, dim=1) + out)
            # out: N x 1 x d_model
            return out, attn_weights

        out = self.attn(x)
        out = self.dropout(out)
        # out = self.layer_norm(out)
        # Experimenting with skip connection
        out = self.layer_norm(torch.mean(x, dim=1) + out)

        return out


class ManyToOneEncoder(nn.Module):
    def __init__(self, layer_embedding_size=5, num_layers=12):
        super().__init__()
        self.d_model = layer_embedding_size * num_layers

        # Create a many to one attention block
        self.attn = ManyToOneAttentionBlock(self.d_model)

        # Get the decoders
        decoder_layer = TransformerDecoderLayer(d_model=self.d_model, nhead=4, dropout=.2)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=12)

        # Embedder and positional encoder
        self.gen_embedder = torch.nn.Embedding(100, 5, padding_idx=0)
        self.pos_encoder = PositionalEncoding(self.d_model)

        # Other items
        self.softmax = torch.nn.Softmax(1)
        self.linear = torch.nn.Linear(layer_embedding_size * num_layers, 10)

    def compute_attention(self, x):
        return self.attn.compute_attention(x)

    def get_decision(self, trajectories, mask, return_attention_weights=False):
        """
        This function is used for inference
        :param trajectories:
        :param mask:
        :param return_attention_weights:
        :return:
        """

        T = len(trajectories)
        # Here T is the sequence length
        x = self.gen_embedder(torch.cat(trajectories)).reshape(T, 1, -1)

        # Get the positional encoding
        x = self.pos_encoder(x)

        # Get the target mask
        tgt_mask = self.decoder.generate_square_subsequent_mask(x.shape[0])

        # Feed through the decoder
        x = self.decoder(x, tgt_mask=tgt_mask)

        # Now reshape to feed through the many-to-one attention block
        x = x.transpose(0,1)
        # Should be 1 x T x d_model

        if return_attention_weights:
            out, attn_weights = self.attn(x, return_attention_weights=return_attention_weights)
            probs = self.softmax(out.view(-1) + mask.unsqueeze(0))
            # out: N x 1 x d_model
            return {"logits": out, "masked_probs": probs, "attention_weights": attn_weights}

        out = self.linear(self.attn(x))
        probs = self.softmax(out.view(-1) + mask.unsqueeze(0))

        return {"logits": out, "masked_probs": probs}

    def normalize_values(self,values,eps=1e-5):
        """
        Implement this if needed
        :return:
        """
        return ((values - torch.min(values,dim=1,keepdim=True).values) / (torch.max(values,dim=1,keepdim=True).values - torch.min(values,dim=1,keepdim=True).values + 1e-5))*2 - 1

    def forward(self, trajectories, return_attention_weights=False, device='cpu'):
        """
        This function is used for training
        Args:
            x ((N x L x d_model) torch.Tensor): the input embeddings
            return_attention_weights (bool): whether to return the tensor weights
                with the network output
        """
        N, T, o = trajectories.shape
        # Here T is the sequence length
        x = self.gen_embedder(trajectories.long()).reshape(N, T, -1)
        x = x.transpose(0,1)

        # Get the positional encoding
        x = self.pos_encoder(x)

        # Get the target mask
        tgt_mask = self.decoder.generate_square_subsequent_mask(x.shape[0])

        # Get the target key padding mask
        tgt_key_padding_mask = None

        # Feed through the decoder
        x = self.decoder(x, tgt_mask=tgt_mask)

        # Now reshape to feed through the many-to-one attention block
        x = x.transpose(0, 1)
        # Should be 1 x T x d_model

        if return_attention_weights:
            out, attn_weights = self.attn(x, return_attention_weights=return_attention_weights)
            # out: N x 1 x d_model
            return {"logits": out, "attention_weights": attn_weights}

        out = self.linear(self.attn(x))

        return {"logits": out}