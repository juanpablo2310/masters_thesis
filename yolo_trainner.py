from ultralytics import YOLO
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--conf_file',default='default',type=str)
args = parser.parse_args()
# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(
                    data=f"{args.conf_file}.yaml", 
                    epochs=10
                    )  # train the modelß