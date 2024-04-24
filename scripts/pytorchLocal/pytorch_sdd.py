import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch.nn as nn
import torch
from tqdm import tqdm
import os
import sys
print(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))
    
from utils_torch import CustomCocoDetection,custom_collate_fn,saveModel
from torchvision.transforms import Resize,Compose
from torch.utils.data import  DataLoader 
from scripts.utils.paths import  get_project_annotations,get_project_data_MELU_dir,get_project_data_UN_dir,get_project_models
import argparse

IMG_SHAPE = (416, 416)

TRANSFORMS = Compose([
    Resize(IMG_SHAPE)
    ])




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

class ResNetFeatures(nn.Module):
    def __init__(self):
        super(ResNetFeatures, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.out_channels = 2048

    def forward(self, x):
        x = self.features(x)
        return x

def correct_boxes(boxes):
    if boxes[2] <= 0 or boxes[3] <= 0:
        boxes[2] = boxes(boxes[2], 1)
        boxes[3] = boxes(boxes[3], 1)
    x1, y1, x2, y2 = boxes
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return [x1, y1, x2, y2]
    
coco_dataset = CustomCocoDetection(IMAGES_FOLDER,COCO_ANNOTATION_FILE,transform=TRANSFORMS)

data_loader = DataLoader(coco_dataset, batch_size=32, shuffle=True,collate_fn = custom_collate_fn )  
# backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
backbone = ResNetFeatures()
out_channels = backbone.out_channels
backbone.out_channels = out_channels


model = FasterRCNN(
    backbone = backbone, 
    num_classes=7,  # 6 object classes + 1 background class
    rpn_anchor_generator=AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)),
    box_detections_per_img=300
)
model = model.to(torch.float32) #float()

model.roi_heads.box_predictor = nn.Linear(out_channels, 7 * 4) 

device = args.device #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = args.epochs

for epoch in tqdm(range(num_epochs),total = num_epochs):
    for images, targets in data_loader:
        images = torch.stack(list(image.permute(2,0,1).to(device) for image in images)) 
        targets = [{k.replace('bbox', 'boxes'): torch.tensor(correct_boxes(v)).unsqueeze(0).to(torch.float32).to(device) if k == 'bbox' else torch.tensor(v).to(torch.float32).to(device) for k, v in t.items() if k in ['bbox','category_id']} for ann in targets for t in ann]
        loss_dict = model(images.to(torch.float32), targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad() 
        losses.backward()
        optimizer.step()
    lr_scheduler.step()

if args.save_model:
    os.makedirs(SAVE_PATH,exist_ok=True)
    fileSaved = os.path.join(SAVE_PATH, f'{args.epochs}_{args.batch_size}_{args.learning_rate}_{args.trainSet}_{args.save_model}')
    saveModel(model,fileSaved)


# torch.save(model.state_dict(), 'model_sdd.pth')

