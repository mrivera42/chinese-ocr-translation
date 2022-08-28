import torch
import json 
import transformer
import numpy as np 
import transformer_training

# load parameters 
with open('transformer_params.json') as fh: 
    params = json.load(fh)
with open('en_index_dict.json') as fh: 
    en_index_dict = json.load(fh)
with open('cn_word_dict.json') as fh: 
    cn_word_dict = json.load(fh)
with open('en_word_dict.json') as fh: 
    en_word_dict = json.load(fh)

# transcription 
transcription = [['請','給','我','一','些','熱','的','東','西','喝','。']]

# add special tokens and padding 
transcription[0].insert(0,'BOS')
transcription[0].append('EOS')
transcription = np.array([np.concatenate([x,['PAD']*(params['d_seq']-len(x))]) if len(x) < params['d_seq'] else x for x in transcription])
print('transcription: ', transcription)

# convert words to ids 
transcription_ids = transformer_training.word_to_id(transcription, cn_word_dict)
print('transcription_ids: ', transcription_ids)

# convert to tensor 
transcription_tensor = torch.Tensor(transcription_ids).to(torch.device("cpu")).long()


# initialize transformer 
transformer = transformer.Transformer(
    d_src_vocab=params['d_src_vocab'],
    d_trg_vocab=params['d_trg_vocab'],
    d_seq=params['d_seq'],
    d_embedding=params['d_embedding'],
    h=params['h'],
    expansion_factor=params['expansion_factor'],
    num_layers=params['num_layers']
)
transformer.load_state_dict(torch.load('models/transformer.pt'))
transformer.to(torch.device("cpu"))
transformer.eval()

# autoregressive input 
# trg = [['BOS']]
trg = [['please', 'give', 'me', 'something', 'hot', 'to', 'drink', '.']]
trg[0].insert(0,'BOS')
trg[0].append('EOS')
trg = np.array([np.concatenate([x,['PAD']*(params['d_seq']-len(x))]) if len(x) < params['d_seq'] else x for x in trg])
trg_ids = transformer_training.word_to_id(trg, en_word_dict)
print('out_ids: ', trg_ids)
trg_tensor = torch.Tensor(trg_ids).to(torch.device('cpu')).long()

# inference 
mask = torch.tril(torch.ones((params['d_seq'], params['d_seq']))).to(torch.device('cpu')).long()
src = transformer.encoder(transcription_tensor)
out1 = transformer.decoder(src, trg_tensor, mask)
val, ind = torch.max(out1,dim=1)
def id_to_word(inds, en_index_dict):

    words = [[en_index_dict[str(ind.item())] for ind in sent] for sent in inds]
    return words
out_sentence = id_to_word(ind, en_index_dict)
print('out_sentence: ', out_sentence)
# # for i in range(params['d_seq'] - 1):
# out = transformer.decoder(memory, start, mask.long())
# val, ind = torch.max(out,dim=1)
# out_sentence = id_to_word(ind, en_index_dict)
# print('out: ', out_sentence)