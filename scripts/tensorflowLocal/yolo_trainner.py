from ultralytics import YOLO
import argparse 
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))
from scripts.utils.paths import get_project_configs,get_project_models


parser = argparse.ArgumentParser(description='trains the YOLOv8 architecture from ultralytics using tensorflow') 
parser.add_argument('--conf_file',default='config_un',type=str)
parser.add_argument('--epochs',type = int, default=5)
parser.add_argument('--batch_size',type = int, default = 10)
parser.add_argument('--resume',type = bool, default = False)
parser.add_argument('--save_model',type = str,default = 'yolov8n.pth')
args = parser.parse_args()

model = YOLO("yolov8n.yaml") 

config_path = get_project_configs(f'yaml/{args.conf_file}.yaml') #os.path.join(get_project_configs(),'yaml' ,f'{args.conf_file}.yaml')
save_path = get_project_models(f'TensorFlow/YOLO/{args.epochs}_{args.batch_size}_{args.learning_rate}_{args.conf_file}_{args.save_model}') #os.path.join(get_project_models(),'TensorFlow','YOLO',f'{args.epochs}_{args.batch_size}_{args.learning_rate}_{args.conf_file}_{args.save_model}')

results = model.train(
                    data=config_path, 
                    name = args.save_model,
                    epochs = args.epochs,
                    batch = args.batch_size,
                    resume = args.resume,
                    project = save_path,
                    device = 'mps'
                    )  # train the modelß