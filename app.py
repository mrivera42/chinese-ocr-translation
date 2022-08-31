from cv2 import cvtColor
import torch 
import pandas as pd 
import cv2
import numpy as np
import sys
import torchvision
import transformer 
import json

from transformer_training import id_to_word, word_to_id

# set device 
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
            "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine.")
else:
    print('MPS available')
    mps_device = torch.device("mps")



im = cv2.imread('example/nihao.jpeg')
im = cv2.resize(im, (416,416))

# YOLOv5 inference 
yolov5 = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5_best.pt',_verbose=False)
results = yolov5(im)  # inference
coordinates = results.pandas().xyxy[0].sort_values('xmin')  # sorted left-right

# crop detected characters left to right 
characters = []
im_copy = im.copy()
for row_index in range(len(coordinates.index)):

    row = coordinates.iloc[row_index]
    xmin = int(row['xmin'])
    xmax = int(row['xmax'])
    ymin = int(row['ymin'])
    ymax = int(row['ymax'])
    character_crop = im_copy[ymin:ymax,xmin:xmax]
    characters.append(character_crop)

character_stack = [cv2.resize(cv2.cvtColor(i,cv2.COLOR_BGR2GRAY),(64,64)) for i in characters]
for i,img in enumerate(character_stack):
    path = f'example/character_{i}.png'
    cv2.imwrite(path=path, im=img)
stack = np.hstack(character_stack)
cv2.imshow("im",stack)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ConvNet classifier 
transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((64,64)),
        torchvision.transforms.Grayscale(1),
        torchvision.transforms.ToTensor(),
    ])
convnet = torch.jit.load('models/model_scripted.pt',map_location='cpu')
convnet.eval() # set dropout layers to eval mode for inference 
classes = np.load('class_names.npy')
transcription = []
for character in  characters:

    im = cv2.cvtColor(character,cv2.COLOR_BGR2GRAY)
    im = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,8)

    im = transforms(im).float()
    im = im.unsqueeze(0) # batch dimension
    logits = convnet(im)
    index = np.argmax(logits.detach().numpy())
    pred = classes[index]
    transcription.append(pred)
print(transcription)
    
# # machine translation 
# with open('transformer_params.json') as fh: 
#     params = json.load(fh)
# transformer = transformer.Transformer(
#     d_src_vocab=params['d_src_vocab'],
#     d_trg_vocab=params['d_trg_vocab'],
#     d_seq=params['d_seq'],
#     d_embedding=params['d_embedding'],
#     h=params['h'],
#     expansion_factor=params['expansion_factor'],
#     num_layers=params['num_layers']
    
# )
# transformer.load_state_dict(torch.load('models/transformer_best.pt'))
# transformer.to(torch.device("cpu"))
# transformer.eval()

# transcription.insert(0,'BOS')
# transcription.append('EOS')
# transcription = [transcription]
# transcription = np.array([np.concatenate([x,['PAD']*(params['d_seq']-len(x))]) if len(x) < params['d_seq'] else x for x in transcription])
# with open('en_index_dict.json') as fh: 
#     en_index_dict = json.load(fh)
# with open('cn_word_dict.json') as fh: 
#     cn_word_dict = json.load(fh)
# transcription = [[cn_word_dict[word] for word in sentence] for sentence in transcription]
# print('transcription: ',transcription)

# transcription_encoder = torch.Tensor(transcription).to(torch.device('cpu')).long()
# print(transcription_encoder.type)
# memory = transformer.encoder(transcription_encoder)
# start_list = [['BOS']]
# start_ind = word_to_id(start_list, cn_word_dict)
# start = torch.Tensor(np.array([np.concatenate([x,[0]* (params['d_seq']-len(x))]) if len(x) < params['d_seq'] else x for x in start_ind])).to(torch.device('cpu')).long()
# mask = torch.tril(torch.ones((params['d_seq'], params['d_seq']))).to(torch.device('cpu')).long()
# # for i in range(params['d_seq'] - 1):
# out = transformer.decoder(memory, start, mask.long())
# val, ind = torch.max(out,dim=1)
# out_sentence = id_to_word(ind, en_index_dict)
# print('out: ', out_sentence)





    




    









    

    
# for index, row in coordinates.iterrows(): 
#     print(row)


