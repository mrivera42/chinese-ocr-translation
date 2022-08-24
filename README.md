# chinese-ocr-webapp

### OCR webapp that uses YOLOv5 to detect chinese characters, ConvNet to classify detected characters, and Transformer network to translate Chinese to English. 

## Character Detection with YOLOv5

## Character Classification 
First I tried a LeNet5 network for 10 epochs and achieved a training accuracy of 71.29% and validation accuracy of 58.61%. There is still a large amount of trianing bias, so the next steps would be to either train for more epochs or increase the size of the network. By looking at the graphs it can be seen that the training accuracy was leveling off, so training for more epochs likely won't do much. So the next step would be to increase the size of the network or modify the architecture in some way. 
Lenet5
Epoch [25/25] train acc: 80.57955806967455 train loss: 0.0024595935828983784 val acc: 66.9184299756736 val loss: inf
Best model acc: 66.9184299756736 epoch: 25

Custom ConNet 
Epoch [20/25] train acc: 87.17038127334364 train loss: 0.0952029898762703 val acc: 74.2592299262224 val loss: inf
Best model acc: 74.2592299262224 epoch: 20

Transformer 
Epoch[100/100] train_loss: 0.009634294547140598 val_loss: 0.03459519321291611
best model - epoch: 100 val loss: 0.03459519321291611




