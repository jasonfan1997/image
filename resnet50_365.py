import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os


arch = 'resnet50'
# load the pre-trained weights
model_weight = 'whole_%s_places365.pth.tar' % arch
if not os.access(model_weight, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/whole_%s_places365.pth.tar' % arch
    os.system('wget ' + weight_url)

useGPU = 0
if useGPU == 1:
    model = torch.load(model_weight)
else:
    model = torch.load(model_weight, map_location=lambda storage, loc: storage) # model trained in GPU could be deployed in CPU machine like this!
    
for param in model.parameters():
    param.requires_grad = False
# Replace the last fully-connected layer
# Parameters of newly constructed modules have requires_grad=True by default
model.fc = torch.nn.Linear(2048, 80)

# Optimize only the classifier
optimizer = torch.optim.Adam(model.fc.parameters())