from tkinter.tix import MAX
import numpy as np
import torch 
from nltk import word_tokenize
import nltk
from collections import Counter 
import matplotlib.pyplot as plt 
import transformer
import matplotlib.pyplot as plt
import json 
from bleu_score import get_bleu_score
import transformer_nlp
import time

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
DEVICE=torch.device("cpu")
DEVICE=torch.device("mps")

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
    total_words = len(ls) 
    word_dict = {w[0]: index  for index,w in enumerate(ls)}
    # inverted index 
    index_dict = {v:k for k,  v in word_dict.items()}
    return word_dict, total_words, index_dict

def wordToID(en, cn, en_dict, cn_dict):
    '''convert word list to id list'''

    out_en_ids = [[en_dict.get(w,0) for w in sent] for sent in en]
    out_cn_ids = [[cn_dict.get(w,0) for w in sent] for sent in cn]

    return out_en_ids, out_cn_ids

def word_to_id(batch, word_dict): 

    out_ids = [[word_dict[word] for word in sent] for sent in batch]
    return out_ids

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

if __name__ == "__main__": 
    start = time.time()

    # load and tokenize data
    train_en, train_cn = load_data(TRAIN_FILE)
    dev_en, dev_cn = load_data(DEV_FILE)


    # build dictionaries 
    en_word_dict, en_total_words, en_index_dict = build_dictionary(train_en)
    cn_word_dict, cn_total_words, cn_index_dict = build_dictionary(train_cn)

    # save english word dict 
    with open("cn_word_dict.json", "w") as fp:
        json.dump(cn_word_dict , fp,ensure_ascii=False) 

    with open("en_index_dict.json", "w") as fp:
        json.dump(en_index_dict , fp) 
    
    with open("en_word_dict.json","w") as fp: 
        json.dump(en_word_dict, fp)

    # use dictionaries to convert each word to an index id 
    train_en, train_cn = wordToID(train_en, train_cn, en_word_dict, cn_word_dict)
    dev_en, dev_cn     = wordToID(dev_en, dev_cn, en_word_dict, cn_word_dict)

    # split into batches 
    train_data = splitBatch(en=train_en, cn=train_cn, batch_size=BATCH_SIZE)
    dev_data = splitBatch(en=dev_en, cn=dev_cn, batch_size=BATCH_SIZE)

    # vocab lengths 
    src_vocab = len(cn_word_dict)
    tgt_vocab = len(en_word_dict)
    print('src_vocab: ', src_vocab)
    print('tgt_vocab: ', tgt_vocab)

    # initialize model 
    model = transformer_nlp.make_model(src_vocab, tgt_vocab)
    # print(model)
    model.to(DEVICE)
    print(model)

    # optimization loop 
    best_loss = 1e5
    best_epoch = 0
    best_bleu = 0
    optimizer=torch.optim.Adam(params=model.parameters(),lr=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    train_losses = []
    val_losses = []
    bleu_scores = []
    for epoch in range(1,EPOCHS+1):

        # train loop 
        for i, (src,tgt) in enumerate(train_data):

            # place tensors to device 
            src = torch.Tensor(src).to(DEVICE).long()
            trg = torch.Tensor(tgt).to(DEVICE).long()

            tgt = trg
            tgt_y = trg

            # tgt = trg[:,:-1] # decoder input (shifted left)
            # tgt_y = trg[:,1:] # decoder target (shifted right)

            # print statements 
            # print('src: ', id_to_word(src, cn_index_dict))
            # print('tgt: ', id_to_word(tgt, en_index_dict))
            # print('tgt_y: ', id_to_word(tgt_y, en_index_dict))
            # print('src shape: ', src.size())
            # print('tgt shape: ', tgt.size())
            # print('tgt_y shape: ', tgt_y.size())

            src_mask = (src != PAD).unsqueeze(-2)
            tgt_mask_ = (tgt != PAD).unsqueeze(-2)
            tgt_mask = tgt_mask_ & transformer_nlp.subsequent_mask(tgt.size(-1)).type_as(tgt_mask_.data)

            # forward pass 
            out = model.forward(src, tgt, src_mask, tgt_mask)

            # print prediction vs target sentence 
            val, ind = torch.max(out, -1)
            pred_sentence = id_to_word(ind,en_index_dict)
            # print('out: ', id_to_word(ind, en_index_dict))
            

            # compute loss 
            train_loss = loss_fn(out.contiguous().view(-1, tgt_vocab), tgt_y.contiguous().view(-1))
            
            # backprop 
            optimizer.zero_grad()
            train_loss.backward()

            # update weights 
            optimizer.step()
        
        val_loss = 0
        num_batches = len(dev_data)
        bleu = 0

        for i, (src, tgt) in enumerate(dev_data):

            # place tensors on device 
            src = torch.Tensor(src).to(DEVICE).long()
            trg = torch.Tensor(tgt).to(DEVICE).long()

            tgt = trg
            tgt_y = trg

            # tgt = trg[:,:-1] # decoder input (shifted left)
            # tgt_y = trg[:,1:] # decoder target (shifted right)

            src_mask = (src != PAD).unsqueeze(-2)
            tgt_mask_ = (tgt != PAD).unsqueeze(-2)
            tgt_mask = tgt_mask_ & transformer_nlp.subsequent_mask(tgt.size(-1)).type_as(tgt_mask_.data)

            # forward pass 
            out = model.forward(src, tgt, src_mask, tgt_mask)
            
            # compute loss 
            loss_val = loss_fn(out.contiguous().view(-1,tgt_vocab),tgt_y.contiguous().view(-1))
            val_loss += loss_val.item()

            # compute bleu score 
            trg_sentences = id_to_word(tgt,en_index_dict)
            # print('trg len: ', len(trg_sentences))
            val, ind = torch.max(out, -1)
            pred_sentences = id_to_word(ind,en_index_dict)
            # print('pred len: ', len(pred_sentences))
            bleu1, bleu2, bleu3, bleu4, bleu_smooth = get_bleu_score(trg_sentences, pred_sentences)
            bleu += bleu4

            
        val_loss /= num_batches
        bleu /= num_batches
        val_losses.append(val_loss)
        train_losses.append(train_loss.item())
        bleu_scores.append(bleu)
        
        

        if val_loss < best_loss:
            best_bleu = bleu
            best_loss = val_loss
            best_epoch = epoch 
            torch.save(model.state_dict(), 'models/transformer.pt')

        print(f'Epoch[{epoch}/{EPOCHS}] train_loss: {train_loss.item()} val_loss: {val_loss} bleu: {bleu}')
    print(f'best model - epoch: {best_epoch} val loss: {best_loss} bleu: {best_bleu}')

    def word_to_id(batch, word_dict):

        out_ids = [[word_dict[word] for word in sent] for sent in batch]
        return out_ids

    plt.plot(train_losses, label='train')
    plt.plot(val_losses,label='val')
    plt.title('Transformer Loss')
    plt.xlabel('Epoch')
    plt.ylabel('CE Loss')
    plt.legend()
    plt.savefig('results/transformer_loss.png')
    plt.close()

    plt.plot(bleu_scores)
    plt.title('Transformer Bleu Score')
    plt.xlabel('Epoch')
    plt.ylabel('Bleu Score')
    plt.savefig('results/transformer_bleu.png')

    end = time.time()

    duration = end - start 
    print('time: ', duration)

    # # init model 
    # # print('d_src_vocab: ',src_vocab)
    # # print('d_trg_vocab: ',tgt_vocab)
    # # print('d_seq: ', MAX_LENGTH)
    # # print('d_embedding: ',D_MODEL)
    # # print('h: ',H_NUM)
    # # print('expansion_factor: ',4)
    # # print('num_layers: ',LAYERS)
    # model_params = {
    #     'd_src_vocab':src_vocab,
    #     'd_trg_vocab':tgt_vocab,
    #     'd_seq':MAX_LENGTH,
    #     'd_embedding':D_MODEL,
    #     'h':H_NUM,
    #     'expansion_factor':4,
    #     'num_layers':LAYERS
    # }
    # with open("transformer_params.json", "w") as fp:
    #     json.dump(model_params , fp)
    
    # model = transformer.Transformer(
    #     d_src_vocab=src_vocab,
    #     d_trg_vocab=tgt_vocab,
    #     d_seq=MAX_LENGTH,
    #     d_embedding=D_MODEL,
    #     h=H_NUM,
    #     expansion_factor=4,
    #     num_layers=LAYERS
    # )
    # model.to(DEVICE)

    # # init mask 


    # # optimization loop 
    # best_loss = 1e5
    # best_epoch = 0
    # best_bleu = 0
    # optimizer=torch.optim.Adam(params=model.parameters(),lr=5e-5)
    # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0) 
    # train_losses = []
    # val_losses = []
    # bleu_scores = []
    # for epoch in range(1,EPOCHS+1):

    #     # train loop 
    #     for i, (src,trg) in enumerate(train_data):

    #         # place tensors to device 
    #         src = torch.Tensor(src).to(DEVICE).long()
    #         trg = torch.Tensor(trg).to(DEVICE).long()
    #         mask = torch.tril(torch.ones((MAX_LENGTH, MAX_LENGTH))).to(DEVICE)

    #         # forward pass 
    #         out = model(src,trg, mask)

    #         # print prediction vs target sentence 
    #         trg_sentence = id_to_word(trg,en_index_dict)

    #         val, ind = torch.max(out, -1)
    #         pred_sentence = id_to_word(ind,en_index_dict)
            

    #         # compute loss 
    #         train_loss = loss_fn(out.view(-1, tgt_vocab), trg.view(-1))
            
    #         # backprop 
    #         optimizer.zero_grad()
    #         train_loss.backward()

    #         # update weights 
    #         optimizer.step()

        
    #     val_loss = 0
    #     num_batches = len(dev_data)
    #     bleu = 0

    #     for i, (src, trg) in enumerate(dev_data):

    #         # place tensors on device 
    #         src = torch.Tensor(src).to(DEVICE).long()
    #         trg = torch.Tensor(trg).to(DEVICE).long()

    #         # forward pass 
    #         out = model(src, trg, mask)
            
    #         # compute loss 
    #         loss_val = loss_fn(out.view(-1,tgt_vocab),trg.view(-1))
    #         val_loss += loss_val.item()

    #         # compute bleu score 
    #         trg_sentences = id_to_word(trg,en_index_dict)
    #         # print('trg len: ', len(trg_sentences))
    #         val, ind = torch.max(out, -1)
    #         pred_sentences = id_to_word(ind,en_index_dict)
    #         # print('pred len: ', len(pred_sentences))
    #         bleu1, bleu2, bleu3, bleu4, bleu_smooth = get_bleu_score(trg_sentences, pred_sentences)
    #         bleu += bleu4

            
    #     val_loss /= num_batches
    #     bleu /= num_batches
    #     val_losses.append(val_loss)
    #     train_losses.append(train_loss.item())
    #     bleu_scores.append(bleu)
        
        

    #     if val_loss < best_loss:
    #         best_bleu = bleu
    #         best_loss = val_loss
    #         best_epoch = epoch 
    #         torch.save(model.state_dict(), 'models/transformer.pt')

    #     print(f'Epoch[{epoch}/{EPOCHS}] train_loss: {train_loss.item()} val_loss: {val_loss} bleu: {bleu}')
    # print(f'best model - epoch: {best_epoch} val loss: {best_loss} bleu: {best_bleu}')

    # def word_to_id(batch, word_dict):

    #     out_ids = [[word_dict[word] for word in sent] for sent in batch]
    #     return out_ids

    # plt.plot(train_losses, label='train')
    # plt.plot(val_losses,label='val')
    # plt.title('Transformer Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('CE Loss')
    # plt.legend()
    # plt.savefig('results/transformer_loss.png')
    # plt.close()

    # plt.plot(bleu_scores)
    # plt.title('Transformer Bleu Score')
    # plt.xlabel('Epoch')
    # plt.ylabel('Bleu Score')
    # plt.savefig('results/transformer_bleu.png')


        







    
