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
#delete this
#image_path ='flowers/test/16/image_06657.jpg'
device='cpu'
parser = argparse.ArgumentParser(
    description='Some arg_pass for predicting',
)
parser.add_argument(type=str, dest='input')
#parser.add_argument(type=str, dest='checkpoint')
parser.add_argument('checkpoint')
parser.add_argument('--gpu', action='store_const', const='gpu')
parser.add_argument('--top_k', dest='top_k',type=int, default=1)
parser.add_argument('--category_names', dest='category_names',type=str, default=None)


args = parser.parse_args()

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

data_dir = 'flowers'
train_dir = data_dir + '/train'
train_dataset = datasets.ImageFolder(train_dir)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,map_location=lambda storage, loc:storage)
    model = models.densenet161(pretrained=True)
    model.classifier = checkpoint['model.classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = train_dataset.class_to_idx    
    return model

def process_image(pil_im):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_im = Image.open(pil_im, 'r')

    pil_im.thumbnail((256,256))
    pil_im=pil_im.crop((16,16,240,240))


    #ish(np.asarray(pil_im))
    np_image = np.array(pil_im)
    
    #couldn't make it the other way
    transformations = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))])
    np_image = transformations(np_image).float()    
    return np_image





def predict(image_path, model, topk=args.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    model.to(device)
    model.eval()
    list_class=[]
    list_probs=[]
    
    with torch.no_grad():
       
        image = process_image(image_path)
        image = image.type(torch.FloatTensor).to(device)   
        
        
        image = image.unsqueeze(0)
        
        logps = model.forward(image)
        ps = torch.exp(logps)
        
        probs_tensor, classes_tensor = ps.topk(topk,dim=1)
       
        
              
            
        for b in probs_tensor[0].cpu().numpy():
            list_probs.append(b)   
            
       
        
        
        for a in classes_tensor[0].cpu().numpy():
            list_class.append(idx_to_class[a])
        
    return list_probs,list_class

  
model=load_checkpoint(args.checkpoint)
idx_to_class = {v: k for k, v in model.class_to_idx.items()}

probs,classes=predict(args.input,model)

    
    



print('\nMOST LIKELY IMAGE CLASS AND PROBABILITY')
classes, probs in zip(classes, probs)
print("Image Class: {}, Probability: {}\n".format(classes[0], probs[0]))
#print("Image Class: {}, Probability: {}".format(names, probs))

if args.top_k >1:
    print('TOP K IMAGE CLASSES AND PROBABILITIES')
    for probs_, classes_ in zip(probs, classes):   
        print("Image Class: {}, Probability: {}".format(classes_, probs_))
    
if args.category_names !=None:
    print('\nIMAGE NAME(S) AND PROBABILITY')
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)    
    names = [cat_to_name[c] for c in classes]  
    for names_map, probs_map in zip(names,probs):  
        print("{}: {}".format(names_map,probs_map))



    





