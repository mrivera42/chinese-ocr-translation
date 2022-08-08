import torch 
import pandas as pd 
import cv2

# load model 
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo_model/best.pt')


# load image for inference 
im = cv2.imread('processed_data/3.jpeg')
im = cv2.resize(im, (416,416))

# inference 
results = model(im)
results = model(im)  # inference
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

for character in characters:
    cv2.imshow("character", character)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    

    
# for index, row in coordinates.iterrows(): 
#     print(row)


