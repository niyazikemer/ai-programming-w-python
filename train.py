import argparse
from PIL import Image
from matplotlib.pyplot import imshow as ish
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import argparse


parser = argparse.ArgumentParser(
    description='Some arg_pass for my project',
)
parser.add_argument(type=str, dest='data_dir')
parser.add_argument('--save_dir', dest='save_dir', type=str, default='')
parser.add_argument('--arch', dest='arch', type=str, default='densenet161')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.002)
parser.add_argument('--hidden_units', dest='hidden_units', type=int, default=1000)
parser.add_argument('--epochs', dest='epochs', type=int, default=1)
parser.add_argument('--gpu', action='store_const',type=str const='gpu')

args = parser.parse_args()


data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'




# TODO: Define your transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_dataset = datasets.ImageFolder(train_dir, transform=test_transforms)



# TODO: Using the image datasets and the trainforms, define the dataloaders

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Use GPU if it's available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    if args.gpu!='gpu':
        print('GPU disabled')
        device='cpu'
    if args.gpu=='gpu':
        print('GPU enabled')
        device='cuda'
else:
    print('GPU is not available, switching CPU')
    device='cpu'
    

hidden_units=args.hidden_units

if args.arch=='densenet161':    
    model = models.densenet161(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(nn.Linear(2208, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, int(hidden_units/2)),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),                                 
                                 nn.Linear(int(hidden_units/2), 102),
                                 nn.LogSoftmax(dim=1))
    
elif args.arch=="vgg16":
    model =models.vgg16(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, int(hidden_units/2)),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),                                 
                                 nn.Linear(int(hidden_units/2), 102),
                                 nn.LogSoftmax(dim=1))


criterion = nn.NLLLoss()
learning_rate=args.learning_rate
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

model.to(device);

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in train_dataloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(valid_dataloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(valid_dataloader):.3f}")
            running_loss = 0
            model.train()

checkpoint = {'model.classifier':model.classifier,
              'model.class_to_idx':train_dataset.class_to_idx,
              'state_dict': model.state_dict(),
              'epoch': epoch,
              'optimizer_state_dict': optimizer.state_dict()  }

if args.save_dir!='':
   torch.save(checkpoint, args.save_dir + '/some.pth')
