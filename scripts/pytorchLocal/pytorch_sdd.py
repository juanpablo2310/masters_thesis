# ⠀⠀⠀⠀⠀⣠⣶⣶⣶⣶⣶⣶⣶⣶⣶⣦⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⣀⣴⣾⠿⠿⠛⠛⠋⠉⠉⠉⠛⠛⠛⠿⢿⣿⣿⣿⣦⣄⠀⠀⠀⠀⠀
# ⠀⠀⣠⣾⠟⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⢿⣿⣿⣷⡄⠀⠀⠀
# ⠰⣶⣿⡏⠀⠀⠀⠀⠀⠀⠀⣠⣶⣶⣦⣄⣀⡀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣄⣀⣀
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⠿⠿⠿⠋⠉⠁⠀⠀⠀⠀⢀⣿⣿⣿⣿⠋⠉⠉
# ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣴⣿⣿⣿⣿⠋⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⢠⣾⣶⣶⣤⣤⣤⣤⣤⣤⣴⣶⣾⣿⣿⣿⣿⠿⠋⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠈⠉⠙⠛⠛⠛⢿⣿⡿⠿⠿⠿⠛⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀
 

import torchvision

import torch
import torch.nn as nn
import torchvision.models.detection
from typing import Iterable
from torchvision.models.detection import SSD300_VGG16_Weights# FasterRCNN,FasterRCNN_ResNet50_FPN_Weights,FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou
import warnings
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from tqdm import tqdm
import os
import pdb
import sys
# print(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))
import pickle
from utils_torch import CustomCocoDetection,custom_collate_fn,saveModel,read_data,calculate_mean_std_per_channel, dict_load, normalize_image
from models_torch import sizeEven, pad_tensors_to_same_size,calculate_iou
from torchvision.transforms import Resize,Compose, ToTensor,Lambda,Normalize
from torch.utils.data import  DataLoader 
from scripts.utils.paths import  get_project_annotations,get_project_data_MELU_dir,get_project_data_UN_dir,get_project_models
import argparse

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.catch_warnings(action='ignore')


class CustomBoxPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomBoxPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


def train(data_loader:Iterable,model:nn.Module,optimizer:torch.optim,lr_scheduler:torch.optim,device:str,num_classes:int):
    for images, targets in data_loader:
        images = torch.stack(list(image.to(device) for image in images))#torch.stack(list(image.permute(2,0,1).to(device) for image in images)) 
        targets = [{k.replace('bbox', 'boxes').replace('category_id','labels'): torch.tensor(bbox_de_COCO_format(v)).unsqueeze(0).to(torch.float32).to(device) if k == 'bbox' else labelVec(int(v),num_classes).to(device) for k, v in t.items() if k in ['bbox','category_id']} for ann in targets for t in ann] #torch.tensor(v).to(torch.int64)
        # targets = [list(x.values()) for x in targets]
        loss_dict = model(images.to(torch.float32), targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad() 
        losses.backward()
        optimizer.step()
    lr_scheduler.step()
    return loss_dict

def test(data_loader:Iterable,model:nn.Module,device:str):
    model.eval()
    with torch.no_grad():
        all_targets, all_preds = [], []
        iou_per_box = []
        for images, targets in data_loader:
            targets_box = [{k.replace('bbox', 'boxes').replace('category_id','labels'): torch.tensor(bbox_de_COCO_format(v)).unsqueeze(0).to(torch.float32).to(device)  for k, v in t.items() if k in ['bbox']} for ann in targets for t in ann]
            targets_box = [torch.stack(list(x.values())) for x in targets_box] #[list(x.values()) for x in targets]
            outputs = model(images.to(torch.float32))
            all_targets.append(targets_box)
            for output in outputs:
                all_preds.append(output['boxes'])
        all_targets_t, all_preds_t = [torch.stack([item for sublist in batch for item in sublist]) for batch in all_targets], torch.stack(all_preds) 
 
        all_targets_t = sizeEven(all_targets_t) 
        all_targets_t,all_preds_t = pad_tensors_to_same_size(torch.stack(all_targets_t),all_preds_t)
        iou_per_box = calculate_iou_batch(all_targets_t,all_preds_t,iou_per_box)
        precision, recall, f1_score = calculate_precision_recall_f1(all_targets_t, all_preds_t)
    return precision, recall, f1_score, iou_per_box

def calculate_precision_recall_f1(targets, preds, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0

    for target, pred in zip(targets, preds):
        iou = calculate_iou(target, pred)
        tp += iou if (iou > iou_threshold) else 0
        fp += iou if (iou <= iou_threshold) else 0
        fn += len(target) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def calculate_map(coco_gt, coco_dt):
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAP = coco_eval.stats[0]
    return mAP


def calculate_iou_batch(target_boxes:torch.Tensor, pred_boxes:torch.Tensor,resultList:list)->torch.Tensor:
    for box1,box2 in zip(pred_boxes,target_boxes):
            iouPerBox = calculate_iou(box1,box2,format=False)
            resultList.append(iouPerBox)
    return resultList

def labelVec(label:int,num_classes:int)->torch.tensor:
    classVec = torch.zeros(num_classes,dtype = torch.int64)
    classVec[label] = 1
    return classVec 
    
    
def bbox_de_COCO_format(bbox:list)->list[float]:
    x_min = bbox[0] - (0.5 * bbox[2])
    y_min = bbox[1] - (0.5 * bbox[3])
    x_max = x_min + bbox[2]
    y_max = y_min + bbox[3]
    return [x_min,y_min,x_max,y_max]   



def main(args:argparse):
    
    if args.trainSet == 'UN':
        IMAGES_FOLDER = os.path.join(get_project_data_UN_dir(),'imagenes')
        COCO_ANNOTATION_FILE = os.path.join(get_project_annotations(),'dataset.json')

    elif args.trainSet == 'MELU':
        IMAGES_FOLDER = os.path.join(get_project_data_MELU_dir(),'train','images')
        COCO_ANNOTATION_FILE = os.path.join(get_project_annotations(),'dataset_MELU.json')
    
    MEAN_VAL, STD_VAL = calculate_mean_std_per_channel(IMAGES_FOLDER)
    IMG_SHAPE = (500, 500) 
    SAVE_PATH = os.path.join(get_project_models(), 'pytorch','SDD')

    TRANSFORMS = Compose([
        Resize(IMG_SHAPE),
        Lambda(lambda x : normalize_image(x,MEAN_VAL,STD_VAL)), 
        ])
    _,_,_,num_classes = read_data(COCO_ANNOTATION_FILE)
    coco_dataset = CustomCocoDetection(IMAGES_FOLDER,COCO_ANNOTATION_FILE,transform=TRANSFORMS) #
    data_loader = DataLoader(coco_dataset, batch_size=32, shuffle=True,collate_fn = custom_collate_fn )  

    model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT) # `weights=SSD300_VGG16_Weights.DEFAULT torchvision.models.detection.retinanet_resnet50_fpn weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT  torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn fasterrcnn_mobilenet_v3_large_fpn
    model = model.to(torch.float32) #float()
    
    device = args.device 
    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    num_epochs = args.epochs

    loss_per_epoch = []
    training_metrics = dict()

    print('starting training ....')
    for epoch in tqdm(range(num_epochs),total = num_epochs):
        loss_dict = train(data_loader=data_loader,model=model,optimizer=optimizer,lr_scheduler=lr_scheduler,device=device,num_classes=num_classes)
        loss_per_epoch.append(loss_dict)
        precision, recall, f1_score, iou_per_box = train(data_loader=data_loader,model=model,device=device) 

        dict_load(training_metrics,epoch,'epoch')
        dict_load(training_metrics,np.mean(iou_per_box),'iou')
        dict_load(training_metrics,precision,'precision')
        dict_load(training_metrics,recall,'recall')
        dict_load(training_metrics,f1_score,'f1Score')
        

    if args.save_model:
        os.makedirs(SAVE_PATH,exist_ok=True)
        fileSaved = os.path.join(SAVE_PATH, f'{args.epochs}_{args.batch_size}_{args.learning_rate}_{args.trainSet}_{args.save_model}')
        loss_per_epoch = [{k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in x.items()} for x in loss_per_epoch]
        total_results = [training_metrics,loss_per_epoch]
        saveModel(model,fileSaved,total_results)
        
def makeArgs():
    parser = argparse.ArgumentParser(description='trains the RestNet50 architecture')
    parser.add_argument('--epochs',type = int, default=5)
    parser.add_argument('--batch_size',type = int, default = 10)
    parser.add_argument('--learning_rate',type = float, default = 0.001)
    parser.add_argument('--save_model',type = str,default = 'model_SDD_pytorch.pth')
    parser.add_argument('--device',type=str,default='cpu')
    parser.add_argument('--trainSet',choices=['UN','MELU'],default = 'UN')
    args = parser.parse_args()
    return args
    
        
if __name__ == '__main__':
    args = makeArgs() 
    main(args)


