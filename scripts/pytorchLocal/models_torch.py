import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Iterable,Callable,List,Dict,Sequence,Union
from torch.nn import functional as F
import pdb
from utils_torch import dict_load
def pad_tensors_to_same_size(tensor1, tensor2):
    diff = tensor1.numel() - tensor2.numel()
    if diff > 0:
        tensor2 = F.pad(tensor2.view(-1), (0, diff))
    elif diff < 0:
        tensor1 = F.pad(tensor1.view(-1), (0, -diff))
    return tensor1, tensor2


def sigmoid(vec:Sequence[Union[int,float]])->Sequence[Union[int,float]]:
    return 1.0 /(1.0 + np.exp(-vec)) #/max(vec) 

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

def joinEntriesOfList(lista:List[any],juntionSize:int = 2):
    jointList = []
    for leftSide in range(0,len(lista),juntionSize):
        rightSide = leftSide + juntionSize
        jointList.append(lista[leftSide:rightSide])
    return jointList



def vectorCategoria(listaCategorias:Sequence[float],totalLabels:int)->Iterable:
    listaVectoresCategorias = []
    for categoriaIndividual in listaCategorias:
        initialVec = [0 for _ in range(totalLabels)] #np.zeros(6) 
        initialVec[int(categoriaIndividual)] = 1
        listaVectoresCategorias.append(initialVec)
    return listaVectoresCategorias

def sizeEven(l:Sequence[any])->Sequence[any]:
    lensl = [len(x) for x in l]
    maxLen = max(lensl)
    for elem in l :
        if len(elem) < maxLen:
            dummyElem = [0 for _ in range(len(elem[0]))]
            while len(elem) < maxLen:
                elem.append(dummyElem)
    return l 

def targetPreparation(target:Iterable,totallabes:int, device:str)->tuple[Sequence[any],Sequence[any]]:
    target_bbox = [] 
    target_category = [] 
    
    for target_batch in target:
        list_batch_bbox = []
        list_batch_category = []

        for x in target_batch:
            # x_bbox = sigmoid(np.array(x['bbox']))
            list_batch_bbox.append(x['bbox'])
            list_batch_category.append(x['category_id'])

        listBatchCategoryVec = vectorCategoria(list_batch_category,totallabes)
        target_bbox.append(list_batch_bbox)
        target_category.append(listBatchCategoryVec)
    
    
    target_category_even = sizeEven(target_category)
    target_bbox_even = sizeEven(target_bbox)
    tensorBboxTargets = torch.from_numpy(np.stack(target_bbox_even))
    tensorCategoryTargets = torch.from_numpy(np.stack(target_category_even))
    tensorBboxTargets = tensorBboxTargets.view(-1)
    tensorCategoryTargets = tensorCategoryTargets.view(-1)
    tensorBboxTargets = tensorBboxTargets.to(torch.float32).to(device)
    tensorCategoryTargets = tensorCategoryTargets.to(torch.float32).to(device)
    return tensorBboxTargets,tensorCategoryTargets
    
def outputPreparation(output:Iterable,tensorCategoryTargets:torch.Tensor,tensorBboxTargets:torch.Tensor)->tuple[Sequence[any],Sequence[any]]:
    bboxOutputsTargets = []
    labelsOutputsTargets = []
    for ctg in range(len(output[1])):
        bboxOutput = output[0][ctg].view(-1)  
        labelsOutput = output[1][ctg].view(-1)
        labelsOutput , tensorCategoryTargets = pad_tensors_to_same_size(labelsOutput,tensorCategoryTargets)
        bboxOutput , tensorBboxTargets = pad_tensors_to_same_size(bboxOutput,tensorBboxTargets)
        bboxOutputsTargets.append((bboxOutput,tensorBboxTargets))
        labelsOutputsTargets.append((labelsOutput,tensorCategoryTargets))
    return bboxOutputsTargets,labelsOutputsTargets

def bbox_de_COCO_format_Tensor(bbox:torch.Tensor)->list[float]:
    x_min = bbox[0] - (0.5 * bbox[2])
    y_min = bbox[1] - (0.5 * bbox[3])
    x_max = x_min + bbox[2]
    y_max = y_min + bbox[3]
    return [x_min.item(),y_min.item(),x_max.item(),y_max.item()]


def calculate_iou(box1:Iterable, box2:Iterable)->float:
    """
    Calculates the IoU (Intersection over Union) between two bounding boxes.
    
    Arguments:
    box1 -- list or tuple containing [x1, y1, x2, y2] coordinates of the first bounding box
    box2 -- list or tuple containing [x1, y1, x2, y2] coordinates of the second bounding box
    
    Returns:
    iou -- float value representing the IoU between the two bounding boxes
    """
  
    box1 = bbox_de_COCO_format_Tensor(bbox=box1)
    box2 = bbox_de_COCO_format_Tensor(bbox=box2)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if (area_box1 == 0) | (area_box2 == 0):
        return 0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    union_area = area_box1 + area_box2 - intersection_area
    
    iou = intersection_area / union_area
   
    return iou

def calculateTotalIOU(network:Callable,totallabes:int, data_loader:Iterable, device:str)->tuple[float,float]:
    #bboxOutputsTargets.append((bboxOutput,tensorBboxTargets))
    network.eval()
    totalIOU = []
 
    with torch.no_grad():
        for data, target in data_loader:
            tensorBboxTargets ,tensorCategoryTargets = targetPreparation(target=target,totallabes=totallabes,device=device)
            
            data = data.to(torch.float32).to(device)
            network = network.to(torch.float32).to(device)
            output = network(data) #float())
            
            bboxOutputsTargets,_ = outputPreparation(output,tensorCategoryTargets,tensorBboxTargets)
            bboxOutputs , tensorBboxTargets = zip(*bboxOutputsTargets)
            for bboxOutput,bboxTarget in zip(bboxOutputs,tensorBboxTargets):
                assert bboxOutput.shape == bboxTarget.shape
                for bbox_ind in range(0,bboxTarget.shape[0],4):
                    singleOutputBox = bboxOutput[bbox_ind:bbox_ind+4]
                    singleTargetBox = bboxTarget[bbox_ind:bbox_ind+4]
                    iou = calculate_iou(singleOutputBox,singleTargetBox)
                    totalIOU.append(iou)
            
    return totalIOU



def bboxLossfn(output:Sequence[any],target:Sequence[any])->float:
    return 1 - (output * target).sum() / (output + target - output * target).sum() #criterion(bboxOutput.long(), tensorBboxTargets.long())
        
def calculateLoss(bboxOutputsTargets:Sequence[tuple],labelsOutputsTargets:Sequence[tuple],criterionLabels:Callable,correctCtg:int = 0,criterionBox:Callable = bboxLossfn)->tuple[Sequence[any],Sequence[any],int]:
   
    labelsOutputs, tensorCategoryTargets = zip(*labelsOutputsTargets)
    bboxOutputs , tensorBboxTargets = zip(*bboxOutputsTargets)
    lossLabelsPerCategory = []
    lossBboxPerCategory = []
    for labelsOutput,tensorCategoryTarget in zip(labelsOutputs,tensorCategoryTargets):
        lossLabels = criterionLabels(labelsOutput, tensorCategoryTarget.to(torch.float32))#float())
        lossLabelsPerCategory.append(lossLabels)
        
        predictedCtg = torch.argmax(labelsOutput)
        targetCtg = torch.argmax(tensorCategoryTarget)
        correctCtg += (predictedCtg == targetCtg).sum().item() 
    
    for bboxOutput,tensorBboxTarget in zip(bboxOutputs,tensorBboxTargets):    
        lossBbox = criterionBox(bboxOutput,tensorBboxTarget)
        lossBboxPerCategory.append(lossBbox)
        
    return lossLabelsPerCategory, lossBboxPerCategory, correctCtg  

def train(network:Callable,totallabes:int, data_loader:Iterable, criterion:Callable, optimizer:Callable, device:str)->tuple[float,Sequence[any],Sequence[any]]:
    network.train()
    running_loss = 0.0

    for data, target in data_loader:

        tensorBboxTargets ,tensorCategoryTargets = targetPreparation(target=target,totallabes=totallabes,device=device)
        
        optimizer.zero_grad()
        data = data.to(torch.float32).to(device)
        network = network.to(torch.float32).to(device)
        output = network(data)#float()) .to(torch.float32)
        
        bboxOutputsTargets,labelsOutputsTargets = outputPreparation(output,tensorCategoryTargets,tensorBboxTargets)
        lossLabelsPerCategory, lossBboxPerCategory,_ = calculateLoss(bboxOutputsTargets,labelsOutputsTargets,criterionLabels = criterion)

        totalLossLabels = sum(lossLabelsPerCategory)
        totalLossBbox = sum(lossBboxPerCategory)
        jointLoss = totalLossBbox + totalLossLabels
        jointLoss.backward()
        optimizer.step()
        running_loss += jointLoss.item() * data.size(0) #.item()
        avg_loss = running_loss / len(data_loader.dataset)

    return avg_loss, lossBboxPerCategory, lossLabelsPerCategory

 
def test(network:Callable,totallabes:int, data_loader:Iterable, criterion:Callable, device:str)->tuple[float,float]:
    network.eval()
    correctCtg = 0
    total = 0
    test_loss = 0.0
 
    with torch.no_grad():
        for data, target in data_loader:
            tensorBboxTargets ,tensorCategoryTargets = targetPreparation(target=target,totallabes=totallabes,device=device)
            
            data = data.to(torch.float32).to(device)
            network = network.to(torch.float32).to(device)
            output = network(data) #float())
            
            bboxOutputsTargets,labelsOutputsTargets = outputPreparation(output,tensorCategoryTargets,tensorBboxTargets)
            lossLabelsPerCategory, lossBboxPerCategory,correctCtg = calculateLoss(bboxOutputsTargets,labelsOutputsTargets,criterionLabels = criterion,correctCtg = correctCtg)
            
            totalLossLabels = sum(lossLabelsPerCategory)
            totalLossBbox = sum(lossBboxPerCategory)
            jointLoss = totalLossBbox + totalLossLabels
            test_loss += jointLoss.item() * data.size(0)
            total += len(target)#.size(0)
  
    return test_loss / len(data_loader.dataset), 100 * correctCtg / total

def labelsOutputsDiscretization(labels:list,total_classes:int)->list:
    
    whole_ctg = []
  
    for single_ctg in labels:
        total_discretization_per_ctg = []
        for i in range(0,len(single_ctg),total_classes):
            
            outputImage = single_ctg[i:i+total_classes]
        
            outputImageDiscrete = np.zeros(total_classes).tolist()
            indentifyCategory = np.argmax(outputImage)
            outputImageDiscrete[indentifyCategory] = 1.0
            
            total_discretization_per_ctg.extend(outputImageDiscrete)
        
        whole_ctg.append(torch.Tensor(total_discretization_per_ctg))
    return whole_ctg

def metricsTotalPerClass(model:Callable, data_loader:Iterable,totallabes:int, device:str)->np.ndarray:

    model.eval()
    total_metrics_per_class = {i:dict() for i in range(totallabes)}
    with torch.no_grad():
        for images, labels in data_loader:
            # Forward pass
            tensorBboxTargets ,tensorCategoryTargets = targetPreparation(target=labels,totallabes=totallabes,device=device)
            images = images.to(torch.float32).to(device)
            output = model(images)
            _,labelsOutputsTargets = outputPreparation(output,tensorCategoryTargets,tensorBboxTargets)
            labelsOutputs, tensorCategoryTargets = zip(*labelsOutputsTargets)
            labelsOutputsDiscrete = labelsOutputsDiscretization(labelsOutputs,total_classes=totallabes)
            class_preds_np = [class_pred.cpu().numpy() for class_pred in labelsOutputsDiscrete]
            labels_np =   [lbels.cpu().numpy() for lbels in tensorCategoryTargets] #tensorCategoryTargets.cpu().numpy()
    

            for dict_categ,ctg_target, ctg_preds in zip(total_metrics_per_class.values(),labels_np, class_preds_np):
                
                assert len(ctg_target) == len(ctg_preds)
                for img_lenght in range(0,len(ctg_target),totallabes):
                    presicion, recall, F1_score   = calculate_metrics(confusion_matrix(ctg_target[img_lenght:img_lenght+totallabes],ctg_preds[img_lenght:img_lenght+totallabes]) , totallabes)
             
                    dict_load(dict_categ,presicion,'precision')
                    dict_load(dict_categ,recall,'recall')
                    dict_load(dict_categ,F1_score,'F1_score')
      
    for dict_ctg in total_metrics_per_class.values():
        for key in dict_ctg.keys():
            dict_ctg[key] = [np.mean(dict_ctg[key])]

    return total_metrics_per_class #confusion_matrix(all_labels, all_predictions) #confusion_mat

def calculate_metrics(cm:np.ndarray,total_ctg:int)->list:  #https://en.wikipedia.org/wiki/Precision_and_recall
    p  = 1
    n = total_ctg - p 
    r_ = n/p
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / p
    tnr = tn / n
    fpr = fp / n 


    presicion = tpr / (tpr + (r_ * fpr)) #tp / (tp + fp)
    recall = tpr
    F1_score = 0 if presicion == recall == 0 else 2 * (presicion * recall / presicion + recall) 

    return presicion, recall, F1_score    



class ModelFromScratch(nn.Module):
    def __init__(self, params: Dict[str, any], num_classes: int, max_n_boxes: int, imageSize: List[int]):
        super(ModelFromScratch, self).__init__()
        self.imageSize = imageSize
        self.params = params
        self.convBlocks = nn.Sequential()
        self.bias = False
        # self.inputLayer = nn.Linear(self.imageSize)

        for i in range(len(self.params['filters']) - 1):
            conv_layer = nn.Conv2d(in_channels=self.params['filters'][i],
                                   out_channels=self.params['filters'][i + 1],
                                   stride=self.params['conv_stride'][i],
                                   kernel_size=self.params['kernel_size_conv'][i],
                                   padding=self.params['conv_padding'][i],
                                   bias=self.bias,
                                   padding_mode=self.params['padding_mode_conv'][i])
            
            self.convBlocks.add_module(f'bl_conv_{i}', conv_layer)

            activation_layer = getActivation(self.params['activation'][i])
            self.convBlocks.add_module(f'bl_conv_act_{i}', activation_layer)

            pool_layer = nn.MaxPool2d(kernel_size=self.params['pool_size'][i],
                                      stride=self.params['pool_stride'][i],
                                      padding=self.params['pool_padding'][i])
            
            self.convBlocks.add_module(f'bl_maxpol_{i}', pool_layer)

        self.flatten = nn.Flatten()

        self.num_classes = num_classes
        self.max_boxes = max_n_boxes

        # Calculate the input size for the output layers
        dummy_input = torch.rand(3, *self.imageSize)
        dummy_output = self.convBlocks(dummy_input)
        output_size = dummy_output.view(-1).size(0)

        # Bounding Box Prediction
        self.bbox_outputs = nn.ModuleList([nn.Sequential(
            nn.Linear(output_size, max_n_boxes * 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(max_n_boxes,  4),
            nn.Sigmoid()  # Output bounding box coordinates between 0 and 1
        ) for _ in range(num_classes)
        ])

        # Class Prediction
        self.class_outputs = nn.ModuleList([nn.Sequential(
            nn.Linear(output_size, max_n_boxes),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(max_n_boxes, num_classes),
            nn.Softmax(dim=1)  # Output class probabilities
        ) for _ in range(num_classes)
        ])

    def forward(self, x):
        x = torch.permute(x,(0,3,1,2))
        x = self.convBlocks(x)
        x = self.flatten(x)
        bbox_preds = [bbox_head(x) for bbox_head in self.bbox_outputs] #self.bbox_outputs(x)
        class_preds = [class_head(x) for class_head in self.class_outputs]#self.class_outputs(x)
        return bbox_preds, class_preds

# class ModelFromScratch(nn.Module):

#     def __init__(self,params:Dict[str,any],num_classes:int,max_n_boxes:int,imageSize:List[int]): #,imagesArray:Iterable params:Dict[str,List[any]],
#         super(ModelFromScratch, self).__init__()
#         self.imageSize = imageSize

#         self.params = params
#         self.convBlocks = nn.Sequential()
#         self.bias = False

#         self.inputLayer = nn.Linear(self.imageSize)

#         for i in range(len(self.params['filters'])-1):
#             conv_layer = nn.Conv2d(in_channels = self.params['filters'][i],out_channels = self.params['filters'][i+1],stride= self.params['conv_stride'][i], kernel_size = self.params['kernel_size_conv'][i], padding= self.params['conv_padding'][i],bias=self.bias) # 
#             self.convBlocks.add_module(f'bl_conv_{i}', conv_layer)
#             activation_layer = getActivation(self.params['activation'][i])  
#             self.convBlocks.add_module(f'bl_conv_act_{i}', activation_layer)
#             pool_layer = nn.MaxPool2d(kernel_size=self.params['pool_size'][i], stride=self.params['pool_stride'][i], padding = self.params['pool_padding'][i] )
#             self.convBlocks.add_module(f'bl_maxpol_{i}', pool_layer)

#         self.flatten = nn.Flatten()
#         self.num_classes = num_classes
#         self.max_boxes = max_n_boxes
    

#         self.bbox_outputs  =  nn.Sequential( #nn.ModuleList([
#                 nn.Linear(9, 128),#(128,128)
#                 getActivation('relu'),
#                 nn.Linear(128, max_n_boxes * 4),
#                 getActivation('relu'),
#             ) #for _ in range(num_classes)
#         # ])

#         self.class_outputs = nn.Sequential( #nn.ModuleList([
#                 nn.Linear(9, 128),
#                 getActivation('relu'),
#                 nn.Linear(128, num_classes),
#                 getActivation('relu'),
#                 nn.Softmax(dim=1),
#             ) #for _ in range(num_classes)
#         # ])

#     def forward(self,x):
#         # x = torch.from_numpy(imageArray)
#         x = self.inputLayer(x)
      
#         # x = x.view(tuple(self.imageSize))
#         x = self.convBlocks(x)
#         #print(x.size())
#         # x = x.view(x.size(0), -1)
#         # x = self.flatten(x)
        
#         # bbox_outputs = self.bbox_outputs(x) #[bbox_head(x) for bbox_head in self.bbox_outputs]
#         # class_outputs = self.class_outputs(x) #[class_head(x) for class_head in self.class_outputs]

#         # bbox_outputs = bbox_outputs.view(-1, self.max_boxes * 4)
#         # class_outputs = class_outputs.view(-1, self.num_classes)


#         # # bbox_outputs = bbox_outputs.view(self.max_boxes,4)#[x.view(-1,4) for x in bbox_outputs]
#         # # class_outputs = class_outputs.view(1,self.num_classes)#[x.view(1,-1) for x in class_outputs]
#         # return bbox_outputs,class_outputs





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
