import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.nn import Module
from torch import Tensor
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
import copy
from typing import Optional, Any


# Altered from Pytorch implementation
class TransformerDecoder(Module):
	r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

	__constants__ = ['norm']

	def __init__(self, decoder_layer, num_layers, norm=None):
		super(TransformerDecoder, self).__init__()
		self.layers = _get_clones(decoder_layer, num_layers)
		self.num_layers = num_layers
		self.norm = norm

	def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
				memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
		r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
		output = tgt

		for mod in self.layers:
			output = mod(output, tgt_mask=tgt_mask,
						 tgt_key_padding_mask=tgt_key_padding_mask)

		if self.norm is not None:
			output = self.norm(output)

		return output

	def generate_square_subsequent_mask(self, sz: int) -> Tensor:
		r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
			Unmasked positions are filled with float(0.0).
		"""
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask


class TransformerDecoderLayer(Module):
	r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
		super(TransformerDecoderLayer, self).__init__()
		self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
		self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
		# Implementation of Feedforward model
		self.linear1 = Linear(d_model, dim_feedforward)
		self.dropout = Dropout(dropout)
		self.linear2 = Linear(dim_feedforward, d_model)

		self.norm1 = LayerNorm(d_model)
		self.norm2 = LayerNorm(d_model)
		self.norm3 = LayerNorm(d_model)
		self.dropout1 = Dropout(dropout)
		self.dropout2 = Dropout(dropout)
		self.dropout3 = Dropout(dropout)

		self.activation = _get_activation_fn(activation)

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = F.relu
		super(TransformerDecoderLayer, self).__setstate__(state)

	def forward(self, tgt: Tensor, tgt_mask: Optional[Tensor] = None,
				tgt_key_padding_mask: Optional[Tensor] = None) -> Tensor:
		r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
		tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
							  key_padding_mask=tgt_key_padding_mask)[0]
		tgt = tgt + self.dropout1(tgt2)
		tgt = self.norm1(tgt)
		# tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
		# 						   key_padding_mask=memory_key_padding_mask)[0]
		# tgt = tgt + self.dropout2(tgt2)
		# tgt = self.norm2(tgt)
		tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
		tgt = tgt + self.dropout3(tgt2)
		tgt = self.norm3(tgt)
		return tgt


def _get_clones(module, N):
	return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
	if activation == "relu":
		return F.relu
	elif activation == "gelu":
		return F.gelu

	raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

class PositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)


class CNNTransformer(nn.Module):
	def __init__(self, dim=128, output_dim=2, n_layers=15, n_classes=10, device='cpu'):
		super().__init__()
		self.d_model = dim
		decoder_layer = TransformerDecoderLayer(d_model=self.d_model, nhead=4, dropout=.2)
		self.decoder = TransformerDecoder(decoder_layer, num_layers=6)
		self.embedder = torch.nn.Embedding(dim+1,dim)
		self.pos_encoder = PositionalEncoding(self.d_model)

		# Hopefully these conv nets will keep track of spatial relationships
		# Can change this later so that it only trains on one cnn network and then I just grab n filters
		# self.down_cnn = {i : nn.Sequential(
		# 					  nn.Conv2d(i, i, 5, 1, 1),
		# 	                  nn.BatchNorm2d(i),
		# 	                  nn.ReLU(True),
		# 	                  nn.Conv2d(i, i, 4, 1, 1),).to(device) for i in range(1,num_quant_channels)}
		# for key in self.down_cnn:
		# 	for param in self.down_cnn[key].parameters():
		# 		param.requires_grad = False
		# self.up_cnn = {i : nn.Sequential(
		# 	nn.ConvTranspose2d(i, i, 4, 1, 1),
		# 	nn.BatchNorm2d(i),
		# 	nn.ReLU(True),
		# 	nn.ConvTranspose2d(i, i, 5, 1, 1),
		# ).to(device) for i in range(1,num_quant_channels+1)}
		# for key in self.down_cnn:
		# 	for param in self.down_cnn[key].parameters():
		# 		param.requires_grad = False
		self.device = device
		self.num_quant_channels = dim

	def forward(self, x):
		# Predict position should be between 1 and dim - 1
		# x is of size B, C, H, W
		predict_position = x.shape[1]
		# print(x.shape)
		# Freeze the conv weights
		# x = self.down_cnn[predict_position](x)

		# b, c, h, w = x.shape
		# x = torch.argmax(x, dim=1) + 1 # 128, 7, 7
		# x = x.view(b, -1).transpose(0,1) # 128, 49
		# x = torch.cat([torch.zeros(128,1).to(self.device),x],dim=0)
		x = self.embedder(x.long().transpose(0,1))

		# x.view(c, b, h*w)
		# x = x.view(c, b, h, w)
		# output is B, d_model, C This is used for the src and target
		# Shift the input to the right
		# x = torch.cat([torch.zeros(1,b,self.d_model).to(self.device),x],dim=0)

		# Injects position information
		x = self.pos_encoder(x)
		tgt_mask = self.decoder.generate_square_subsequent_mask(x.shape[0]).to(self.device)
		x = self.decoder(x, tgt_mask=tgt_mask)
		# output is B, d_model, 1
		# We want B, H, W, C again
		x = x.transpose(0, 1)
		# print(x.shape)
		# Freeze the conv weights
		# x = self.up_cnn[predict_position+1](x)
		# output is B, H, W, 2 this is compared with our target with cross entropy.
		return x

	def generate(self, shape=(8, 8), batch_size=100):
		with torch.no_grad():
			x = torch.zeros(
				(batch_size, 1),
				dtype=torch.int64, device=self.device
			)

			for i in range(shape[0]*shape[1]):
				x_old = x.clone()
				x = self.forward(x)
				probs = torch.softmax(x, dim=-1)
				m = torch.distributions.Categorical(probs[:,-1,:])
				x = torch.cat([x_old, m.sample().reshape(-1,1)],dim=1)
				# x = torch.cat([torch.zeros(batch_size, 1).to(self.device), x], dim=1)

			# for i in range(shape[0]):
			# 	for j in range(shape[1]):
			# 		logits = self.forward(x, label)
			# 		probs = F.softmax(logits[:, :, i, j], -1)
			# 		x.data[:, i, j].copy_(
			# 			probs.multinomial(1).squeeze().data
			# 		)
			return x[:,1:]


if __name__ == '__main__':
	cnn_transformer = CNNTransformer(dim=16)
	src = torch.rand((1, 1, 7, 7))
	out = cnn_transformer(src)
	print(out.shape) # Do I train on the last output or all the outputs?