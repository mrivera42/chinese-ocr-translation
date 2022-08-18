from cv2 import cvtColor
import torch 
import pandas as pd 
import cv2
from classifier_training import Model
import numpy as np

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

# load model 
yolov5 = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5_best.pt',_verbose=False)


# load image for inference 
im = cv2.imread('processed_data/28.jpeg')
im = cv2.resize(im, (416,416))

# inference 
results = yolov5(im)
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

# for character in characters:
#     cv2.imshow("character", character)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
character_stack = [cv2.resize(cv2.cvtColor(i,cv2.COLOR_BGR2GRAY),(64,64)) for i in characters]

# classifier 
convnet = torch.jit.load('models/model_scripted.pt',map_location='cpu')
convnet.eval() # set dropout layers to eval mode for inference 
classes = np.load('class_names.npy')
transcription = ''
for character in  characters:
    gray = cv2.cvtColor(character,cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(gray, (64,64))
    reshape = np.reshape(resize,(1,1,64,64))
    tensor = torch.tensor(reshape).float()
    probabilities = torch.nn.functional.softmax(convnet(tensor),dim=1) # get softmax probabilities 
    value, index = torch.max(probabilities,dim=1) # get index of max probabilites
    ind = index[0].item() # get index number from tensor
    pred = classes[index]
    transcription += pred
print(transcription)
    
stack = np.hstack(character_stack)
cv2.imshow("im",stack)
cv2.waitKey(0)
cv2.destroyAllWindows()








    

    
# for index, row in coordinates.iterrows(): 
#     print(row)


