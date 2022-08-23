from tkinter.tix import MAX
import numpy as np
import torch 
from nltk import word_tokenize
from collections import Counter 
import matplotlib.pyplot as plt 
import transformer
import matplotlib.pyplot as plt
import json 

# INIT PARAMETERS 
UNK = 0 # unknown word id 
PAD = 1 # padding word id 
BATCH_SIZE = 64 
EPOCHS=20
LAYERS=3
H_NUM=8
D_MODEL=128
D_FF=256
DROPOUT=0.1
MAX_LENGTH=60
TRAIN_FILE='/Users/maxrivera/Desktop/chinese-english-dataset/train.txt'
DEV_FILE='/Users/maxrivera/Desktop/chinese-english-dataset/dev.txt'
SAVE_FILE='models/transformer_model.pt'
# set device 
DEVICE=torch.device("mps")
DEVICE=torch.device("cpu")




# DATA PREPROCESSING 
def seq_padding(X, padding=0):
    ''' padd each sentence with zeros up to the max length of the entire dataset '''
    return np.array([np.concatenate([x,[padding]*(MAX_LENGTH-len(x))]) if len(x) < MAX_LENGTH else x for x in X])
    
def load_data(path):
    ''' tokenize the data '''
    en = []
    cn = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            en.append(["BOS"] + word_tokenize(line[0].lower(),language='english') + ["EOS"])
            cn.append(["BOS"] + word_tokenize(" ".join([w for w in line[1]])) + ["EOS"])
    return en, cn 

def build_dictionary(sentences, max_words=50000):
    '''
    key(word): value(id)
    key(id): value(word)
    '''

    # sort unique words by frequency    
    word_count = Counter()
    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1

    # index of each word corresponds to its order in the frequency list 
    ls = word_count.most_common(max_words)
    ls.insert(0,('UNK',1))
    ls.insert(0,('PAD',0))
    # print('ls: ',ls)
    total_words = len(ls) 
    word_dict = {w[0]: index  for index,w in enumerate(ls)}
    # print('word_dict: ',word_dict)


    # inverted index 
    index_dict = {v:k for k,  v in word_dict.items()}
    # print('index_dict: ',index_dict)
    return word_dict, total_words, index_dict

def wordToID(en, cn, en_dict, cn_dict):
    '''convert word list to id list'''

    out_en_ids = [[en_dict.get(w,0) for w in sent] for sent in en]
    out_cn_ids = [[cn_dict.get(w,0) for w in sent] for sent in cn]

    return out_en_ids, out_cn_ids

def id_to_word(batch, index_dict):
    '''
    Converts a batch of english and chinese id data to words 
    Args:
    - batch: batch_size x seq_len 
    - index_dict: key(id) -> value(word)
    Returns: 
    - out_words: batch_size x seq_len 
    '''
    out_words = [[index_dict[index.item()] for index in sent] for sent in batch]

    return out_words


def splitBatch(en, cn, batch_size, shuffle=True):
    ''' split dataset into batches '''

    # a list with the start index of each batch 
    idx_list = np.arange(0,len(en),batch_size)

    # shuffle the start indices 
    if shuffle:
        np.random.shuffle(idx_list)
    
    # get all the indices for each batch start index
    batch_indexs = []
    for idx in idx_list:
        batch_indexs.append(np.arange(idx,min((idx + batch_size, len(en)))))

    # use the indices to get the data 
    batches = []
    for batch_index in batch_indexs:
        batch_en = [en[index] for index in batch_index]
        batch_cn = [cn[index] for index in batch_index]

        # add paddings 
        
        batch_en = seq_padding(batch_en)
        batch_cn = seq_padding(batch_cn)

        # append each batch 
        batches.append((batch_cn, batch_en))
    
    return batches 

# load and tokenize data
# print('TOKENIZE DATA')
train_en, train_cn = load_data(TRAIN_FILE)
dev_en, dev_cn = load_data(DEV_FILE)
# print('dev_en: ', dev_en)
# print('dev_cn: ',dev_cn)


# build dictionaries 
en_word_dict, en_total_words, en_index_dict = build_dictionary(train_en)
cn_word_dict, cn_total_words, cn_index_dict = build_dictionary(train_cn)

# save english word dict 
with open("cn_word_dict.json", "w") as fp:
    json.dump(cn_word_dict , fp,ensure_ascii=False) 

with open("en_index_dict.json", "w") as fp:
    json.dump(en_index_dict , fp) 

# print('en total words: ',en_total_words)
# print('cn total words: ', cn_total_words)
# use dictionaries to convert each word to an index id 
train_en, train_cn = wordToID(train_en, train_cn, en_word_dict, cn_word_dict)
dev_en, dev_cn     = wordToID(dev_en, dev_cn, en_word_dict, cn_word_dict)

# print('max value in train_cn: ', np.max([np.max(sent) for sent in train_cn]))
# print('max value in dev_cn: ', np.max([np.max(sent) for sent in dev_cn]))
# split into batches 
train_data = splitBatch(en=train_en, cn=train_cn, batch_size=BATCH_SIZE)
dev_data = splitBatch(en=dev_en, cn=dev_cn, batch_size=BATCH_SIZE)

# vocab lengths 
src_vocab = len(cn_word_dict)
tgt_vocab = len(en_word_dict)
# print(f'src_vocab: {src_vocab}')
# print(f'tgt_vocab: {tgt_vocab}')

# init model 
# print('d_src_vocab: ',src_vocab)
# print('d_trg_vocab: ',tgt_vocab)
# print('d_seq: ', MAX_LENGTH)
# print('d_embedding: ',D_MODEL)
# print('h: ',H_NUM)
# print('expansion_factor: ',4)
# print('num_layers: ',LAYERS)
model_params = {
    'd_src_vocab':src_vocab,
    'd_trg_vocab':tgt_vocab,
    'd_seq':MAX_LENGTH,
    'd_embedding':D_MODEL,
    'h':H_NUM,
    'expansion_factor':4,
    'num_layers':LAYERS
}
with open("transformer_params.json", "w") as fp:
    json.dump(model_params , fp)
 
model = transformer.Transformer(
    d_src_vocab=src_vocab,
    d_trg_vocab=tgt_vocab,
    d_seq=MAX_LENGTH,
    d_embedding=D_MODEL,
    h=H_NUM,
    expansion_factor=4,
    num_layers=LAYERS
)
model.to(DEVICE)

# init mask 
mask = torch.tril(torch.ones((MAX_LENGTH, MAX_LENGTH))).to(DEVICE)

# optimization loop 
best_loss = 1e5
best_epoch = 0
optimizer=torch.optim.Adam(params=model.parameters(),lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0) 
train_losses = []
val_losses = []
for epoch in range(1,EPOCHS+1):

    # train loop 
    for i, (src,trg) in enumerate(train_data):

        # place tensors to device 
        src = torch.Tensor(src).to(DEVICE).long()
        trg = torch.Tensor(trg).to(DEVICE).long()

        # forward pass 
        out = model(src,trg, mask)
        # print('out: ', out.size())
        # print('trg: ', trg.size())
        # print(trg)
        # print('out reshaped: ', out.view(-1, tgt_vocab).size())
        # print('trg reshaped: ', trg.view(-1).size())


        # compute loss 
        train_loss = loss_fn(out.view(-1, tgt_vocab), trg.view(-1))
        
        # backprop 
        optimizer.zero_grad()
        train_loss.backward()

        # update weights 
        optimizer.step()

        # convert to words 
        trg_words = id_to_word(trg,en_index_dict)
        # print('label: ',trg_words)
        val, ind = torch.max(out,dim=-1)
        # print('ind: ',ind)
        out_words = id_to_word(ind,en_index_dict)
        # print('predict: ',out_words)
    
    val_loss = 0
    num_batches = len(dev_data)

    for i, (src, trg) in enumerate(dev_data):

        # place tensors on device 
        src = torch.Tensor(src).to(DEVICE).long()
        trg = torch.Tensor(trg).to(DEVICE).long()

        # forward pass 
        out = model(src, trg, mask)

        # print('out: ', out.size())
        # print('trg: ',trg.size())
        # print('out reshape: ', out.view(-1, tgt_vocab).size())
        # print('trg reshape: ', trg.view(-1).size())
        
        # compute loss 
        loss_val = loss_fn(out.view(-1,tgt_vocab),trg.view(-1))
        val_loss += loss_val.item()
        
    val_loss /= num_batches
    val_losses.append(val_loss)
    train_losses.append(train_loss.item())
    

    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), 'models/transformer.pt')

    print(f'Epoch[{epoch}/{EPOCHS}] train_loss: {train_loss.item()} val_loss: {val_loss}')
print(f'best model - epoch: {best_epoch} val loss: {best_loss}')

def word_to_id(batch, word_dict):

    out_ids = [[word_dict[word] for word in sent] for sent in batch]
    return out_ids

# input = [['She','married','him','.']]
# input_ids = word_to_id(input, cn_word_dict)


plt.plot(train_losses, label='train')
plt.plot(val_losses,label='val')
plt.title('Transformer Loss')
plt.xlabel('epoch')
plt.ylabel('CE Loss')
plt.legend()
plt.savefig('results/transformer_loss.png')


        







    
