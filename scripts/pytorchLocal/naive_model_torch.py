import numpy as np
import argparse
# from pickle import load
from json import load,dump
from tqdm import tqdm
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))

import torch
from torch import nn,optim
from torchinfo import summary
from torch.utils.data import  DataLoader
from torchvision.transforms import Resize,Compose,Normalize
import pdb

from utils_torch import get_maximum_number_of_annotation_in_set,read_data,\
    CustomCocoDetection,custom_collate_fn,saveModel

from models_torch import ModelFromScratch,train,test,metricsTotalPerClass,calculateTotalIOU,calculate_metrics
from scripts.utils.paths import get_project_annotations,get_project_data_MELU_dir,get_project_data_UN_dir,get_project_models,get_project_configs

parser = argparse.ArgumentParser(description='trains the custom architecture')
parser.add_argument('--epochs',type = int, default=5)
parser.add_argument('--batch_size',type = int, default = 10)
parser.add_argument('--learning_rate',type = float, default = 0.001)
parser.add_argument('--model_structure',type = str, default = get_project_configs('json/torch_simple.json'))
parser.add_argument('--save_model',type = str,default = 'model_CNN_pytorch.pth')
parser.add_argument('--trainSet',choices=['UN','MELU'],default = 'UN')
parser.add_argument('--device',type=str,default='cpu')
args = parser.parse_args()

IMAGES_FOLDER = get_project_data_UN_dir('imagenes') if args.trainSet == 'UN' else get_project_data_MELU_dir('train/images/')  #os.path.join(get_project_data_UN_dir(),'imagenes')
COCO_ANNOTATION_FILE = get_project_annotations('dataset.json') if args.trainSet == 'UN' else get_project_annotations('dataset_MELU.json')
SAVE_PATH = get_project_models('pytorch/CNN/') #os.path.join(get_project_models(), 'pytorch','CNN')

# MEAN_VAL, STD_VAL = calculate_mean_std_per_channel(IMAGES_FOLDER)


IMG_SHAPE = (500, 500)
TRANSFORMS = Compose([
    Resize(IMG_SHAPE),
    # Normalize(mean = MEAN_VAL, std = STD_VAL)
    ])

config_object = open(args.model_structure,'rb')
network_structure = load(config_object)

images,annotations,c,num_classes = read_data(COCO_ANNOTATION_FILE)



max_n_boxes = get_maximum_number_of_annotation_in_set(annotations,images) 
coco_dataset = CustomCocoDetection(IMAGES_FOLDER,COCO_ANNOTATION_FILE,transform=TRANSFORMS)
data_loader = DataLoader(coco_dataset, batch_size = args.batch_size, shuffle=True,collate_fn = custom_collate_fn )


num_epochs = args.epochs
learning_rate = args.learning_rate

  
BasicModel = ModelFromScratch(network_structure,num_classes,max_n_boxes,IMG_SHAPE)
criterion = nn.functional.cross_entropy #nn.CrossEntropyLoss()
optimizer = optim.Adam(BasicModel.parameters(), lr=learning_rate)
 
train_loss_history_avg = []
train_loss_history_bbox = []
train_loss_history_lbs = []
test_loss_history = []
test_accuracy_history = []
confusion_matrix_history = []
presicion_history =  {i:list() for i in range(num_classes)}
recall_history = {i:list() for i in range(num_classes)}
F1_score_history = {i:list() for i in range(num_classes)}
iou_history = []

 
for epoch in tqdm(range(num_epochs), total=num_epochs):

    train_loss_avg,train_loss_bbox,train_loss_lbs = train(BasicModel,num_classes, data_loader, criterion, optimizer, args.device)
    test_loss, test_accuracy = test(BasicModel,num_classes, data_loader, criterion, args.device)
    dict_metric_per_epoch = metricsTotalPerClass(BasicModel,data_loader,num_classes,args.device)
    iou = calculateTotalIOU(BasicModel,num_classes, data_loader, args.device)
    #precision, recall, F1_score = calculate_metrics(cm)
    
  

    train_loss_history_avg.append(train_loss_avg)
    train_loss_history_bbox.append([x.item() for x in train_loss_bbox])
    train_loss_history_lbs.append([x.item() for x in train_loss_lbs])
    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_accuracy)
    iou_history.append(iou)

for key in dict_metric_per_epoch:
    presicion_history[key].extend(dict_metric_per_epoch[key]['precision'])
    recall_history[key].extend(dict_metric_per_epoch[key]['recall'])
    F1_score_history[key].extend(dict_metric_per_epoch[key]['F1_score'])    



results = {
'train_loss_history': {'average_loss':train_loss_history_avg,'loss_bbox_per_class':train_loss_history_bbox,'loss_class_per_class':train_loss_history_lbs},
'test_loss_history': test_loss_history,
'test_accuracy_history': test_accuracy_history,
'Intersection_over_union' : iou_history,
'presicion' : presicion_history,
'recall': recall_history,
'F1 Score': F1_score_history,
}

print(results)



if args.save_model:
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    fileSaved = os.path.join(SAVE_PATH, f'{args.epochs}_{args.batch_size}_{args.learning_rate}_{args.trainSet}_{args.save_model}')
    saveModel(BasicModel,fileSaved,results)
   
    