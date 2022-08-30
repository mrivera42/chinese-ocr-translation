from tkinter.tix import MAX
import torch
import json 
import transformer
import numpy as np 
import transformer_training
from bleu_score import get_bleu_score


# evaluate transfomer 
# INIT PARAMETERS 
UNK = 0 # unknown word id 
PAD = 1 # padding word id 
BATCH_SIZE = 64 
EPOCHS=100
LAYERS=3
H_NUM=8
D_MODEL=128
D_FF=256
DROPOUT=0.1
MAX_LENGTH=60
TRAIN_FILE='/Users/maxrivera/Desktop/chinese-english-dataset/train.txt'
DEV_FILE='/Users/maxrivera/Desktop/chinese-english-dataset/dev.txt'
SAVE_FILE='models/transformer_model.pt'
DEVICE=torch.device("cpu")
# load and tokenize data
train_en, train_cn = transformer_training.load_data(TRAIN_FILE)
dev_en, dev_cn = transformer_training.load_data(DEV_FILE)

# build dictionaries 
en_word_dict, en_total_words, en_index_dict = transformer_training.build_dictionary(train_en)
cn_word_dict, cn_total_words, cn_index_dict = transformer_training.build_dictionary(train_cn)

# use dictionaries to convert each word to an index id 
train_en, train_cn = transformer_training.wordToID(train_en, train_cn, en_word_dict, cn_word_dict)
dev_en, dev_cn     = transformer_training.wordToID(dev_en, dev_cn, en_word_dict, cn_word_dict)

# split into batches 
train_data = transformer_training.splitBatch(en=train_en, cn=train_cn, batch_size=BATCH_SIZE)
dev_data = transformer_training.splitBatch(en=dev_en, cn=dev_cn, batch_size=BATCH_SIZE)

# vocab lengths 
src_vocab = len(cn_word_dict)
tgt_vocab = len(en_word_dict)


# initialize model 
model = transformer.Transformer(
        d_src_vocab=src_vocab,
        d_trg_vocab=tgt_vocab,
        d_seq=MAX_LENGTH,
        d_embedding=D_MODEL,
        h=H_NUM,
        expansion_factor=4,
        num_layers=LAYERS
    )
model.load_state_dict(torch.load('models/transformer.pt'))
model.to(torch.device("cpu"))
model.eval()

val_loss = 0
num_batches = len(dev_data)
bleu = 0
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0) 

for i, (src, trg) in enumerate(dev_data):

    # place tensors on device 
    src = torch.Tensor([src[0]]).to(DEVICE).long()
    trg = torch.Tensor([trg[0]]).to(DEVICE).long()
    mask = torch.tril(torch.ones((MAX_LENGTH, MAX_LENGTH))).to(DEVICE)

    # forward pass 
    out = model(src, trg, mask)
    
    # compute loss 
    loss_val = loss_fn(out.view(-1,tgt_vocab),trg.view(-1))
    val_loss += loss_val.item()

    # compute bleu score 
    trg_sentences = transformer_training.id_to_word(trg,en_index_dict)
    print('trg: ', trg_sentences)
    val, ind = torch.max(out, -1)
    pred_sentences = transformer_training.id_to_word(ind,en_index_dict)
    print('pred: ', pred_sentences)
    bleu1, bleu2, bleu3, bleu4, bleu_smooth = get_bleu_score(trg_sentences, pred_sentences)
    bleu += bleu4

    
val_loss /= num_batches
bleu /= num_batches

print(f'val_loss: {val_loss} bleu_score: {bleu}')

# load parameters 
with open('transformer_params.json') as fh: 
    params = json.load(fh)
with open('en_index_dict.json') as fh: 
    en_index_dict = json.load(fh)
with open('cn_word_dict.json') as fh: 
    cn_word_dict = json.load(fh)
with open('en_word_dict.json') as fh: 
    en_word_dict = json.load(fh)

# PREPARE SRC  
transcription = [['他','昨','天','來','我','的','辦','公','室','。']]
# add special tokens and padding 
transcription[0].insert(0,'BOS')
transcription[0].append('EOS')
transcription = np.array([np.concatenate([x,['PAD']*(params['d_seq']-len(x))]) if len(x) < params['d_seq'] else x for x in transcription]) # padding
print('src sentence: ', transcription)
# convert words to ids 
transcription_ids = transformer_training.word_to_id(transcription, cn_word_dict)
# convert to tensor 
transcription_tensor = torch.Tensor(transcription_ids).to(DEVICE).long()

# NON-AUTOREGRESSIVE INPUT
print('----- NON-AUTOREGRESSIVE INPUT -----')
trg = [['he', 'came', 'to', 'my', 'office', 'yesterday','.']]
trg[0].insert(0,'BOS')
trg[0].append('EOS')
trg = np.array([np.concatenate([x,['PAD']*(MAX_LENGTH-len(x))]) if len(x) < MAX_LENGTH else x for x in trg])
print('trg sentence: ', trg)
trg_ids = transformer_training.word_to_id(trg, en_word_dict) # convert target sentence to ids 
trg_tensor = torch.Tensor(trg_ids).to(torch.device('cpu')).long() # convert target ids to tensor 
mask = torch.tril(torch.ones((MAX_LENGTH, MAX_LENGTH))).to(DEVICE).long() # create mask 
out = model(transcription_tensor, trg_tensor, mask)
val, ind = torch.max(out,dim=-1)
def id_to_word(inds, en_index_dict):
    words = [[en_index_dict[str(ind.item())] for ind in sent] for sent in inds]
    return words
out_sentence = id_to_word(ind, en_index_dict)
print('out_sentence: ', out_sentence)

# AUTOREGRESSIVE INPUT 
print('--- AUTOREGRESSIVE INPUT-----')
mask = torch.tril(torch.ones((MAX_LENGTH, MAX_LENGTH))).to(DEVICE).long()
trg = [['BOS']]
trg = np.array([np.concatenate([x,['BOS']*(MAX_LENGTH-len(x))]) if len(x) < MAX_LENGTH else x for x in trg])
print('trg at time=0: ',trg)
trg_ids = transformer_training.word_to_id(trg, en_word_dict)
trg_tensor = torch.Tensor(trg_ids).to(DEVICE).long()
src = model.encoder(transcription_tensor)
for i in range(1,MAX_LENGTH):

    out = model.decoder(src, trg_tensor, mask)
    val, ind = torch.max(out, dim=-1)
    trg_tensor[0][i] = ind[0][i]
    # print(f'trg_tensor at time {i}: ', trg_tensor)

out_sentence = id_to_word(trg_tensor, en_index_dict)
print('out_sentence: ', out_sentence)

# val, ind = torch.max(out, dim=-1)
# out_sentence = id_to_word(ind, en_index_dict)
# print('out_sentencee: ', out_sentence)



