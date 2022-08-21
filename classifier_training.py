import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time


class CustomModel(torch.nn.Module): 

    def __init__(self): 
        super(CustomModel, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1,32,kernel_size=(5,5),stride=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=(2,2),stride=2),
            torch.nn.Conv2d(32,64,kernel_size=(5,5),stride=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=(2,2),stride=2),
            torch.nn.Conv2d(64,128, kernel_size=(5,5),stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(), 
            torch.nn.AvgPool2d(kernel_size=(2,2),stride=2),
            torch.nn.Flatten()
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(4*4*128,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,103)
        )
        
    
    def forward(self, inputs): 
        x = self.features(inputs)
        x = self.classifier(x)

        return x 

class LeNet5(torch.nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.convolutional_layer = torch.nn.Sequential(            
            torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.Conv2d(in_channels=16, out_channels=128, kernel_size=5, stride=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )

        self.linear_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=120, out_features=120),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=120, out_features=103),
        )

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = self.linear_layer(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


if __name__ == "__main__": 

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

    # define transforms 
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64,64)),
        torchvision.transforms.Grayscale(1),
        torchvision.transforms.ToTensor()
    ])

    # load dataset 
    path = '/Users/maxrivera/Desktop/chinese-character-dataset'
    trainpath = path + '/top100/Train'
    testpath = path + '/top100/Test'
    ds_train = torchvision.datasets.ImageFolder(
        root=trainpath,
        transform=transforms
    )
    ds_test = torchvision.datasets.ImageFolder(
        root=testpath,
        transform=transforms
    )

    classes = ds_test.classes 
    np.save("class_names",np.array(classes))
    print('num_classes: ',len(np.array(classes)))
    # print(f'ds len: {len(ds)}')
    # print(f'num classes: {len(ds.classes)}')

    # take subset of whole dataset 
    # subset_len = int(.2 * len(ds))
    # ds_subset = torch.utils.data.Subset(ds,np.arange(subset_len))
    # print(f'subset len: {len(ds_subset)}')
    # print(ds_subset[0][0].shape)
    

    # # split into test and train 
    # train_len = int(subset_len * 0.9)
    # ds_train = torch.utils.data.Subset(ds_subset,np.arange(train_len))
    # ds_test = torch.utils.data.Subset(ds_subset, np.arange(train_len,subset_len))

    print(f'new_train_len: {len(ds_train)}')
    print(f'new_test_len: {len(ds_test)}')
    

    # create dataloaders 
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=64,shuffle=True,num_workers=4,pin_memory=True)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=64,shuffle=True,num_workers=4,pin_memory=True)

    # LENET5
    # model = CustomModel()
    model = CustomModel()
    model.to(mps_device)
    print(model)

    
    # # model = torchvision.models.mobilenet_v3_small(pretrained=False)
    # model = torchvision.models.mobilenet_v2(weights=torchvision.models.mobilenetv2.MobileNet_V2_Weights)
    # model.classifier[1] = torch.nn.Linear(1280,7330)
    # for param in model.features.parameters(): 
    #     param.requires_grad = False
    # model.to(mps_device)
    # print(model)

    # train loop 
    lr = 1e-3
    batch_size = 64
    epochs = 30
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    metrics = {
        'training_acc': [],
        'training_loss': [],
        'val_acc': [],
        'val_loss': []
    }

    start_time = time.time()

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1,epochs+1): 

        num_correct = 0
        total = 0


        for batch_idx, (x,y) in enumerate(dl_train): 
            
            # tensors to device 
            x = x.to(mps_device)
            y = y.to(mps_device)

            # forward
            pred = model(x)

            # compute loss 
            loss = loss_fn(pred, y)

            # backprop 
            optimizer.zero_grad()
            loss.backward()

            # update weights
            optimizer.step()

            # update metrics
            num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += len(x)

            if batch_idx + 1 % 10 == 0:
                print(f'[{total}/{len(ds_train)}] train acc: {num_correct / total * 100}')


        training_loss = loss.item()
        training_acc = num_correct / total * 100 
            

        # val loop 
        num_correct = 0
        total = 0 
        val_loss = 0
        with torch.no_grad(): 

            for x,y in dl_test: 

                # tensors to device
                x = x.to(mps_device)
                y = y.to(mps_device)

                # forward
                pred = model(x)

                # loss 
                loss = loss_fn(pred, y)

                # update metrics 
                num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                total += len(x)
                val_loss += loss.item()
            

        val_acc = num_correct / total * 100 
        val_loss /= len(dl_test)

        
    
        print(f'Epoch [{epoch}/{epochs}] train acc: {training_acc} train loss: {training_loss} val acc: {val_acc} val loss: {val_loss}')
        metrics['training_acc'].append(training_acc)
        metrics['training_loss'].append(training_loss)
        metrics['val_acc'].append(val_acc)
        metrics['val_loss'].append(val_loss)

        # save model with best validation accuracy 
        if val_acc > best_acc: 
            best_acc = val_acc
            best_epoch = epoch 
            model_scripted = torch.jit.script(model) # Export to TorchScript
            model_scripted.save('models/model_scripted.pt')

        # update the learning rate scheduler 
        # scheduler.step()
    print(f'Best model acc: {best_acc} epoch: {best_epoch}')

    # time duration 
    end_time = time.time()
    duration = end_time - start_time
    print(f'Time per epoch: {duration}')
    # plot metrics 
    plt.plot(metrics['training_acc'],label='training_acc')
    plt.plot(metrics['val_acc'],label='val_acc')
    plt.title('classifier accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('results/classifier_acc.png')
    plt.close()

    plt.plot(metrics['training_loss'],label='training_loss')
    plt.plot(metrics['val_loss'],label='val_loss')
    
    plt.title('classifier loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('results/classifier_loss.png')
    plt.close()









