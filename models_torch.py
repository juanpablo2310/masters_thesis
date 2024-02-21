import torch
import torch.nn as nn
from typing import Iterable,Callable,List,Dict


def getActivation(function:str)->Callable:

    function_dict = {
        'relu' : nn.ReLU(),
        'softmax' : nn.Softmax(),
        'sigmoid' : nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leakyReLU': nn.LeakyReLU()
    }
    
    if not function in list(function_dict.keys()):
        raise AttributeError('Please choose valid activation function')
    else:
        return function_dict[function]


 
# def train(network:Callable, data_loader:Iterable, criterion:Callable, optimizer:Callable, device:str):
#     network.train()
#     running_loss = 0.0
 
#     for data, target in data_loader:
#         data, target = data.to(device), target.to(device)
#         data = data.view(data.shape[0], -1)
 
#         optimizer.zero_grad()
#         output = network(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
 
#         running_loss += loss.item() * data.size(0)
 
#     return running_loss / len(data_loader.dataset)

def train(network:Callable, data_loader:Iterable, criterion:Callable, optimizer:Callable, device:str):
    network.train()
    running_loss = 0.0
 
    for data_chunk in data_loader:
        # print(data_chunk)
        data,target = data_chunk[0],data_chunk[0:]
        # data, target = data.to(device), target.to(device)
        #data = data.view(data.shape[0], -1)
 
        optimizer.zero_grad()
        output = network(data)
        print(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
 
        running_loss += loss.item() * data.size(0)
 
    return running_loss / len(data_loader.dataset)
 
def test(network:Callable, data_loader:Iterable, criterion:Callable, device:str):
    network.eval()
    correct = 0
    total = 0
    test_loss = 0.0
 
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)
 
            output = network(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
 
    return test_loss / len(data_loader.dataset), 100 * correct / total


class ModelFromScratch(nn.Module):

    def __init__(self,params:Dict[str,List[any]],normalization_value:float,num_classes:int,max_n_boxes:int,imageSize:List[int]): #,imagesArray:Iterable
        super(ModelFromScratch, self).__init__()
        
        self.imageSize = imageSize
        self.params = params
        self.convBlocks = nn.Sequential()

        for i in range(len(params['filters'])-1):
            conv_layer = nn.Conv2d(in_channels = params['filters'][i],out_channels = params['filters'][i+1], kernel_size = params['kernel_size'][i], padding= 0 if params['conv_padding'][i] == 'same' else 1 ) 
            self.convBlocks.add_module(f'bl_conv_{i}', conv_layer)
            activation_layer = getActivation(params['activation'][i])  
            self.convBlocks.add_module(f'bl_conv_act_{i}', activation_layer)
            pool_layer = nn.MaxPool2d(params['pool_size'][i], stride=params['stride'][i], padding = 0 if params['pool_padding'][i] == 'same' else 1 )
            self.convBlocks.add_module(f'bl_maxpol_{i}', pool_layer)

        self.flatten = nn.Flatten()
        self.num_classes = num_classes
        self.max_boxes = max_n_boxes
        self.scalefactor = nn.Parameter(torch.tensor(1/normalization_value))

        self.bbox_outputs  =  nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                getActivation('relu'),
                nn.Linear(128, max_n_boxes * 4),
                getActivation('relu'),
            ) for _ in range(num_classes)
        ])

        self.class_outputs =  nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 128),
                getActivation('relu'),
                nn.Linear(128, num_classes),
                getActivation('relu'),
                nn.Softmax(dim=1),
            ) for _ in range(num_classes)
        ])

    def forward(self,imageArray):
        x = torch.from_numpy(imageArray)
        # x = self.inputLayer(x)
        x = self.scalefactor * x 

        x = self.convBlocks(x)
        x = self.flatten(x)
        
        bbox_outputs = [bbox_head(x) for bbox_head in self.bbox_outputs]
        class_outputs = [class_head(x) for class_head in self.class_outputs]

        bbox_outputs = [x.view(self.max_boxes,-1) for x in bbox_outputs]
        class_outputs = [x.view(self.num_classes,-1) for x in class_outputs]
        return bbox_outputs,class_outputs





# import torch.optim as optim
# from torchmetrics import IoU
# from torchvision import models
# import torch.nn.functional as F

# class CustomIoUMetric(nn.Module):
#     def __init__(self):
#         super(CustomIoUMetric, self).__init__()
#         self.intersection = nn.Parameter(torch.zeros(1))
#         self.union = nn.Parameter(torch.zeros(1))

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_true = torch.clamp(y_true, 0, 1)
#         y_pred = torch.clamp(y_pred, 0, 1)
#         intersection = torch.sum(torch.minimum(y_true, y_pred))
#         union = torch.sum(torch.maximum(y_true, y_pred))
#         self.intersection.data.add_(intersection)
#         self.union.data.add_(union)

#     def forward(self):
#         return self.intersection / (self.union + torch.finfo(torch.float32).eps)

#     def reset_state(self):
#         self.intersection.data.zero_()
#         self.union.data.zero_()

# class Rescaling(nn.Module):
#     def __init__(self, scale):
#         super(Rescaling, self).__init__()
#         self.scale = nn.Parameter(torch.tensor(scale))
#     def forward(self, x):
#         return x * self.scale


# class ModelBackbone(nn.Module):
#     def __init__(self, params, normalization_value):
#         super(ModelBackbone, self).__init__()
#         self.rescaling = Rescaling(1.0 / normalization_value)
#         self.conv_blocks = nn.Sequential()
#         for i in range(len(params['filters'])-1):
#             conv_layer = nn.Conv2d(in_channels = params['kernel_size'][i],out_channels = params['kernel_size'][i+1], kernel_size = params['kernel_size'][i], padding= 0 if params['conv_padding'][i] == 'same' else 1 ) 
#             #self.conv_blocks.add_module(f'bl_conv_{i}', conv_layer)
#             activation_layer = getActivation(params['activation'][i])  
#             self.conv_blocks.add_module(f'bl_conv_act_{i}', activation_layer(conv_layer))
#             pool_layer = nn.MaxPool2d(params['pool_size'][i], stride=params['stride'][i], padding = 0 if params['pool_padding'][i] == 'same' else 1 )
#             self.conv_blocks.add_module(f'bl_maxpol_{i}', pool_layer)
#         self.flatten = nn.Flatten()
#     def forward(self, x):
#         x = self.rescaling(x)
#         x = self.conv_blocks(x)
#         x = self.flatten(x)
#         return x
    

# class ModelOutputs(nn.Module):
#     def __init__(self, num_classes, max_n_boxes):
#         super(ModelOutputs, self).__init__()
#         self.num_classes = num_classes
#         self.max_boxes = max_n_boxes
#         self.bbox_outputs = nn.ModuleList([
#             nn.Sequential(
#                 getActivation('relu')(nn.Linear(128, 128)),
#                 getActivation('relu')(nn.Linear(128, max_n_boxes * 4)),
#             ) for _ in range(num_classes)
#         ])
        # self.class_outputs = nn.ModuleList([
        #     nn.Sequential(
        #         getActivation('relu')(nn.Linear(128, 128)),
        #         getActivation('relu')(nn.Linear(128, num_classes)),
        #         nn.Softmax(dim=1),
        #     ) for _ in range(num_classes)
#         ])

#     def forward(self, x):
#         bbox_outputs = [bbox_head(x) for bbox_head in self.bbox_outputs] #.view(self.max_boxes, 4)
#         class_outputs = [class_head(x) for class_head in self.class_outputs] #.view(1, self.num_classes)
#         bbox_outputs = [x.view(self.max_boxes,-1) for x in  bbox_outputs] #.view(self.max_boxes, 4)
#         class_outputs = [x.view(self.num_classes,-1) for x in  class_outputs]
#         return bbox_outputs, class_outputs

# class ModelConsolidation(nn.Module):
#     def __init__(self, num_classes):
#         super(ModelConsolidation, self).__init__()
#         self.num_classes = num_classes
#         self.losses = {
#             f'class_head{i}': nn.CrossEntropyLoss() for i in range(num_classes)
#         }
#         self.losses.update({f'bbox_head{i}': nn.MSELoss() for i in range(num_classes)})

#     def forward(self, bbox_outputs, class_outputs):
#         outputs = bbox_outputs + class_outputs
#         loss_functions = self.losses
#         return outputs,loss_functions
    
# class ModelInputs(nn.Module):

#     def __init__(self,imageSize):
#         super(ModelInputs, self).__init__()
#         self.imageSize = imageSize
    
#     def forward(self,images_array):
#         x = torch.from_numpy(images_array)
#         return x
