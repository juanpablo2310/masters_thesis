from ultralytics import YOLO
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--conf_file',default='default',type=str)
parser.add_argument('--epochs',default=10,type=int)
parser.add_argument('--batch',default=5,type=int)

args = parser.parse_args()
# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(
                    data=f"{args.conf_file}.yaml", 
                    epochs=args.epochs,
                    batch = args.batch,
                    device = 'mps'
                    )  # train the modelß