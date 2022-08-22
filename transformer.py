import torch
import math 
import numpy as np



class Embedding(torch.nn.Module):

	def __init__(self, d_vocab, d_embedding):

		super(Embedding, self).__init__()
		# print('d_vocab: ',d_vocab)
		# print('d_embedding: ',d_embedding)
		self.embedding = torch.nn.Embedding(d_vocab, d_embedding)

	def forward(self,x):

		# print('embedding in: ',x.size())
		# print('max value in embedding: ', torch.max(x))

		x = self.embedding(x)
		# print('embedding out: ',x.size())
		return x 


class PositionalEmbedding(torch.nn.Module):

	def __init__(self, d_seq, d_embedding):

		super(PositionalEmbedding, self).__init__()

		self.d_embedding = d_embedding
		self.d_seq = d_seq 

		pe = torch.zeros((d_seq, d_embedding))

		for pos in range(d_seq):

			for i in range(0,d_embedding,2):

				pe[pos, i] = math.sin(pos / (10000 ** ((2*i)/d_embedding)))
				pe[pos, i+1] = math.cos(pos / (10000 ** ((2*(i+1))/d_embedding)))

		pe = pe.unsqueeze(0) # add batch dimension
		self.register_buffer('pe',pe)

	def forward(self, x):
		# print('pe in: ',x.size())



		# seq_len = x.size(1)

		# make embedding relatively larger 
		x *= math.sqrt(self.d_embedding)

		# add positional encoding 
		x += self.pe[:,:self.d_seq]

		# print('pe out: ',x.size())
		return x 


class MultiHeadAttention(torch.nn.Module):

	def __init__(self, d_seq, d_embedding, h):

		super(MultiHeadAttention, self).__init__()

		self.d_seq = d_seq
		self.d_embedding = d_embedding 
		self.h = h
		self.d_k = int(self.d_embedding / self.h)
		self.w_q  = torch.nn.Linear(self.d_k, self.d_k)
		self.w_k = torch.nn.Linear(self.d_k, self.d_k)
		self.w_v = torch.nn.Linear(self.d_k, self.d_k)
		self.w_o = torch.nn.Linear(self.d_embedding,self.d_embedding)


	def forward(self,key, query, value, mask=None):

		# get batch dimension
		batch_size = key.size(0)

		# split into h heads 
		key = key.view(batch_size, self.d_seq, self.h, self.d_k)
		query = query.view(batch_size, self.d_seq,self.h,self.d_k)
		value = value.view(batch_size, self.d_seq, self.h, self.d_k)

		# linear 
		k = self.w_k(key)
		q = self.w_q(query)
		v = self.w_v(value)

		# reshape before multiplying 
		k = k.transpose(1,2)
		q = q.transpose(1,2)
		v = v.transpose(1,2)

		# matmul 
		product = q @ k.transpose(-1,-2)

		# scale 
		product /= math.sqrt(self.d_k)

		# mask 
		if mask is not None:
			product.masked_fill(mask==0,float('-1e20'))

		# softmax 
		product = torch.nn.functional.softmax(product,dim=1)

		# matmul
		score = product @ v 

		# concat
		score = score.transpose(1,2)
		concat = score.contiguous().view(batch_size, self.d_seq, self.d_embedding)

		# linear
		out = self.w_o(concat)

		return out

class TransformerBlock(torch.nn.Module):

	def __init__(self, d_seq, d_embedding, h, expansion_factor):

		super(TransformerBlock, self).__init__()

		self.attention = MultiHeadAttention(d_seq, d_embedding, h)
		self.norm1 = torch.nn.LayerNorm(d_embedding)
		self.feed_forward = torch.nn.Sequential(
			torch.nn.Linear(d_embedding, d_embedding * expansion_factor),
			torch.nn.ReLU(),
			torch.nn.Linear(d_embedding * expansion_factor, d_embedding))
		self.norm2 = torch.nn.LayerNorm(d_embedding)

	def forward(self, key, query, value, mask=None):

		# multihead attention
		sublayer1 = self.attention(key, query, value)
		out1 = self.norm1(sublayer1 + value)
		sublayer2 = self.feed_forward(out1)
		out2 = self.norm2(sublayer2 + out1)
		return out2

class Encoder(torch.nn.Module):

	def __init__(self, d_vocab, d_seq, d_embedding, h, expansion_factor, num_layers):

		super(Encoder, self).__init__()
		# print('-- Inside Encoder Init --')
		self.embedding = Embedding(d_vocab, d_embedding)
		self.positionalembedding = PositionalEmbedding(d_seq, d_embedding)
		self.layers = torch.nn.ModuleList([TransformerBlock(d_seq, d_embedding, h, expansion_factor) for i in range(num_layers)])


	def forward(self, x):
		# print('-- Inside Encoder Forward --')

		x = self.embedding(x)
		x = self.positionalembedding(x)
		for layer in self.layers:

			x = layer(x, x, x)

		return x 


class DecoderBlock(torch.nn.Module):

	def __init__(self, d_seq, d_embedding, h, expansion_factor):

		super(DecoderBlock, self).__init__()
		self.attention = MultiHeadAttention(d_seq, d_embedding, h)
		self.norm = torch.nn.LayerNorm(d_embedding)
		self.transformerblock = TransformerBlock(d_seq, d_embedding, h, expansion_factor)

	def forward(self, enc_out, x, mask):

		attention = self.attention(enc_out, enc_out, x, mask)
		out1 = self.norm(attention + x)
		out2 = self.transformerblock(out1, out1, out1)


		return out2


class Decoder(torch.nn.Module):

	def __init__(self, d_vocab, d_seq, d_embedding, h, expansion_factor, num_layers):

		super(Decoder, self).__init__()
		# print('-- Inside Decoder Init --')

		self.embedding = Embedding(d_vocab, d_embedding)
		self.positionalembedding = PositionalEmbedding(d_seq, d_embedding)
		self.layers = torch.nn.ModuleList([DecoderBlock(d_seq, d_embedding, h, expansion_factor) for i in range(num_layers)])
		self.fc_out = torch.nn.Linear(d_embedding,d_vocab)

	def forward(self, enc_out, x, mask):

		# print('-- Inside Decoder Forward --')

		x = self.embedding(x)
		x = self.positionalembedding(x)
		for layer in self.layers:

			x = layer(enc_out, x, mask)

		# x = torch.nn.functional.softmax(self.fc_out(x),dim=1)
		x = self.fc_out(x)
		return x


class Transformer(torch.nn.Module):

	def __init__(self, d_src_vocab,d_trg_vocab, d_seq, d_embedding, h, expansion_factor,num_layers):

		super(Transformer, self).__init__()

		self.encoder = Encoder(d_src_vocab,d_seq,d_embedding, h, expansion_factor,num_layers)
		self.decoder = Decoder(d_trg_vocab, d_seq, d_embedding, h, expansion_factor,num_layers)

	# def create_mask(self,trg):

	# 	d_seq = trg.size(1)
	# 	mask = torch.tril(torch.ones((d_seq, d_seq)))
		
	# 	return mask
	
	def forward(self, src,trg, mask):

		# mask = self.create_mask(trg)

		x = self.encoder(src)
		x = self.decoder(x, trg, mask)
		return x










		























