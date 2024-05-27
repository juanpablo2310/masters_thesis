import torchvision

import torch
import torch.nn as nn
import torchvision.models.detection

from torchvision.models.detection import FasterRCNN,FasterRCNN_ResNet50_FPN_Weights,FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tqdm import tqdm
import os
import pdb
import sys
# print(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))
    
from utils_torch import CustomCocoDetection,custom_collate_fn,saveModel,read_data,calculate_mean_std_per_channel
from torchvision.transforms import Resize,Compose, ToTensor,Lambda
from torch.utils.data import  DataLoader 
from scripts.utils.paths import  get_project_annotations,get_project_data_MELU_dir,get_project_data_UN_dir,get_project_models
import argparse

parser = argparse.ArgumentParser(description='trains the RestNet50 architecture')
parser.add_argument('--epochs',type = int, default=5)
parser.add_argument('--batch_size',type = int, default = 10)
parser.add_argument('--learning_rate',type = float, default = 0.001)
parser.add_argument('--save_model',type = str,default = 'model_SDD_pytorch.pth')
parser.add_argument('--device',type=str,default='cpu')
parser.add_argument('--trainSet',choices=['UN','MELU'],default = 'UN')
args = parser.parse_args()

if args.trainSet == 'UN':
    IMAGES_FOLDER = os.path.join(get_project_data_UN_dir(),'imagenes')
    COCO_ANNOTATION_FILE = os.path.join(get_project_annotations(),'dataset.json')

elif args.trainSet == 'MELU':
    IMAGES_FOLDER = os.path.join(get_project_data_MELU_dir(),'train','images')
    COCO_ANNOTATION_FILE = os.path.join(get_project_annotations(),'dataset_MELU.json')


SAVE_PATH = os.path.join(get_project_models(), 'pytorch','SDD')

class CustomBoxPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomBoxPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


def calculate_precision_recall_f1(targets, preds, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0

    for target, pred in zip(targets, preds):
        iou = calculate_iou(target, pred)
        tp += (iou > iou_threshold).sum().item()
        fp += (iou <= iou_threshold).sum().item()
        fn += len(target) - (iou > iou_threshold).sum().item()

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


def calculate_iou(target_boxes, pred_boxes):
    iou = box_iou(target_boxes, pred_boxes)
    return iou


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

def normalize_image(image, mean, std):
    image = torch.tensor(image).view(-1, 1, 1)
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return (image - mean) / std 

MEAN_VAL, STD_VAL = calculate_mean_std_per_channel(IMAGES_FOLDER)
IMG_SHAPE = (500, 500) 

TRANSFORMS = Compose([
    Resize(IMG_SHAPE),
    #ToTensor(),
    Lambda(lambda x : normalize_image(x,MEAN_VAL,STD_VAL))    
    ])


_,_,_,num_classes = read_data(COCO_ANNOTATION_FILE)


coco_dataset = CustomCocoDetection(IMAGES_FOLDER,COCO_ANNOTATION_FILE,transform=TRANSFORMS)
data_loader = DataLoader(coco_dataset, batch_size=32, shuffle=True,collate_fn = custom_collate_fn )  

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
model = model.to(torch.float32) #float()
backbone_out_channels = 1024  
model.roi_heads.box_predictor = CustomBoxPredictor(backbone_out_channels, num_classes + 1)

device = args.device 
model = model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = args.epochs

loss_per_epoch = []

for epoch in tqdm(range(num_epochs),total = num_epochs):
    for images, targets in data_loader:
        images = torch.stack(list(image.permute(2,0,1).to(device) for image in images)) 
        targets = [{k.replace('bbox', 'boxes').replace('category_id','labels'): torch.tensor(bbox_de_COCO_format(v)).unsqueeze(0).to(torch.float32).to(device) if k == 'bbox' else labelVec(int(v),num_classes).to(device) for k, v in t.items() if k in ['bbox','category_id']} for ann in targets for t in ann] #torch.tensor(v).to(torch.int64)
        loss_dict = model(images.to(torch.float32), targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad() 
        losses.backward()
        optimizer.step()
    lr_scheduler.step()
    model.eval()
    with torch.no_grad():
        all_targets, all_preds = [], []
        for images, targets in data_loader:
            targets = [{k.replace('bbox', 'boxes').replace('category_id','labels'): torch.tensor(bbox_de_COCO_format(v)).unsqueeze(0).to(torch.float32).to(device) if k == 'bbox' else labelVec(int(v),num_classes).to(device) for k, v in t.items() if k in ['bbox','category_id']} for ann in targets for t in ann]
            outputs = model(images.to(torch.float32))
            all_targets.extend(targets)
            all_preds.extend(outputs)

        iou = calculate_iou(all_targets, all_preds)
        precision, recall, f1_score = calculate_precision_recall_f1(all_targets, all_preds)
        # # If using COCO format
        # mAP = calculate_map(coco_gt, coco_dt)

        print(f'Epoch {epoch}: IoU {iou.mean()}, Precision {precision}, Recall {recall}, F1 Score {f1_score}') # mAP {mAP}'
    loss_dict['epoch'] = epoch
    loss_dict['iou'] = iou.mean()
    loss_dict['precision'] = precision
    loss_dict['recall'] = recall
    loss_dict['f1Score'] = f1_score
    loss_per_epoch.append(loss_dict)

if args.save_model:
    os.makedirs(SAVE_PATH,exist_ok=True)
    fileSaved = os.path.join(SAVE_PATH, f'{args.epochs}_{args.batch_size}_{args.learning_rate}_{args.trainSet}_{args.save_model}')
    loss_per_epoch = [{k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in x.items()} for x in loss_per_epoch]
    saveModel(model,fileSaved,loss_per_epoch)


