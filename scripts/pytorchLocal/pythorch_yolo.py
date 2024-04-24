
from ultralytics import YOLO

import sys
print(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))

import torch.nn as nn
import torch
import os
from tqdm import tqdm 
import numpy as np
import torch.nn.functional as F
from scripts.utils.paths import get_project_configs,get_project_models
import argparse

parser = argparse.ArgumentParser(description='trains the YOLOv8 architecture from ultralytics') 
parser.add_argument('--conf_file',default='config_un',type=str)
parser.add_argument('--epochs',type = int, default=5)
parser.add_argument('--batch_size',type = int, default = 10)
parser.add_argument('--resume',type = bool, default = False)
parser.add_argument('--save_model',type = str,default = 'yolov8n.pth')
args = parser.parse_args()

config_path = get_project_configs(f'yaml/{args.conf_file}.yaml') #os.path.join(get_project_configs() ,'yaml', f'{args.conf_file}.yaml')
save_path = get_project_models(f'pytorch/YOLO/{args.epochs}_{args.batch_size}_{args.learning_rate}_{args.conf_file}_{args.save_model}') #os.path.join(get_project_models(),'pytorch','YOLO',f'{args.epochs}_{args.batch_size}_{args.learning_rate}_{args.conf_file}_{args.save_model}')

model = YOLO('yolov8n.yaml') 
device = torch.device('mps' if torch.has_mps else 'cpu')




results = model.train(
                    data = config_path , #f"config_melu.yaml", }
                    name = args.save_model,
                    epochs = args.epochs,
                    batch = args.batch_size,
                    device = device,
                    resume = args.resume,
                    project = save_path,
                    )  # train the modelß





