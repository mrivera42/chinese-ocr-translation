import torch 
import torchvision


if __name__ == "__main__":


    # load model 
    model = torch.jit.load('models/model_scripted.pt',map_location='cpu')
    model.eval()




    # define transforms 
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64,64)),
        torchvision.transforms.Grayscale(1),
        torchvision.transforms.ToTensor()
    ])

    # load dataset 
    path = '/Users/maxrivera/Desktop/chinese-character-dataset'
    trainpath = path + '/CASIA-HWDB_Train/Train'
    testpath = path + '/top100/Test'

    ds_test = torchvision.datasets.ImageFolder(
        root=testpath,
        transform=transforms
    )

    print(f'test_len: {len(ds_test)}')


    # create dataloaders 
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=64,shuffle=True,num_workers=4,pin_memory=True)


    # val loop 
    loss_fn = torch.nn.CrossEntropyLoss()
    num_correct = 0
    total = 0 
    val_loss = 0
    with torch.no_grad(): 

        for x,y in dl_test: 

            # tensors to device
            x = x.to('cpu')
            y = y.to('cpu')

            # forward
            pred = model(x)
            prob = torch.nn.functional.softmax(pred)
            value, index = torch.max(prob,1)
            ind = index[0].item() 
            print(ind)
            print(y)

            # loss 
            loss = loss_fn(pred, y)

            # update metrics 
            num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += len(x)
            val_loss += loss.item()
        

    val_acc = num_correct / total * 100 
    val_loss /= len(dl_test)

    print(f'val_acc: {val_acc}')