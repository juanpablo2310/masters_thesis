import tensorflow as tf
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))

import numpy as np
import argparse

# from pickle import load
from json import load

from utils_tensorflow import get_maximun_number_of_annotation_in_set, train_batch_consolidation\
    ,image_train_set,image_test_set,read_data,dump_file,saveModel
from models import model_inputs,model_backbone,model_outputs,model_consolidation
from scripts.utils.paths import get_project_models,get_project_configs,get_project_annotations

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='trains the custom architecture')
parser.add_argument('--epochs',type = int, default=5)
parser.add_argument('--batch_size',type = int, default = 10)
parser.add_argument('--model_structure',type = str, default = get_project_configs('json/default_config.json'))
parser.add_argument('--save_model',type = str,default = 'model_CNN_tensorflow.keras')
parser.add_argument('--device',type=str,default='cpu')
parser.add_argument('--learning_rate',type = float, default = 0.001)
parser.add_argument('--trainSet',choices=['UN','MELU'],default = 'UN')
args = parser.parse_args()


IMG_SHAPE = (500, 500)


COCO_ANNOTATION_FILE = get_project_annotations('dataset.json') if args.trainSet == 'UN' else get_project_annotations('dataset_MELU.json') #os.path.join(get_project_annotations,'dataset.json') if args.trainSet == 'UN' else os.path.join(get_project_annotations,'dataset_MELU.json')

SAVE_PATH = get_project_models('TensorFlow/CNN') #os.path.join(get_project_models(), 'TensorFlow','CNN')
SETS_PATHS = get_project_configs('sets') #os.path.join(get_project_configs(),'sets')
os.makedirs(SETS_PATHS,exist_ok=True)

config_object = open(args.model_structure,'rb')
network_structure = load(config_object)

images,annotations,_,num_classes = read_data(COCO_ANNOTATION_FILE)

train_images = image_train_set(images,0.75)
dump_file(train_images,os.path.join(SETS_PATHS,'train_image_set'))
test_images = image_test_set(images,train_images)
dump_file(test_images,os.path.join(SETS_PATHS,'test_image_set'))

total_training_images = [] # 'x'
total_training_targets = [] #'y'

max_n_boxes = get_maximun_number_of_annotation_in_set(annotations,images)  # calculo el numero maximo de cajas por categoria sobre el set total de muestras para garantizar que todas las instancias van a ser detectadas

images_for_training , training_targets = train_batch_consolidation(
                                                                train_images,
                                                                total_training_images,
                                                                annotations,
                                                                total_training_targets,
                                                                max_n_boxes,
                                                                IMG_SHAPE,
                                                                num_classes
                                                                )

normalization_value = np.max([x[0][0] for x in images_for_training])


layer_inputs = model_inputs(IMG_SHAPE)
layer_inputs.shape
feature_extraction = model_backbone(layer_inputs,network_structure,normalization_value)
bbox_outs, class_outs = model_outputs(feature_extraction,num_classes,max_n_boxes)
base_model = model_consolidation(num_classes,layer_inputs,bbox_outs,class_outs)

images_for_training = tf.cast(images_for_training, dtype=tf.float32)

with tf.device(f'/{args.device}:0'):
    history = base_model.fit(
                        images_for_training, training_targets,
                        batch_size = args.batch_size,
                        epochs = args.epochs,
                        shuffle=True,
                        verbose=0
                        )

if args.save_model:
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    saveModel(base_model,os.path.join(SAVE_PATH , f'{args.epochs}_{args.batch_size}_{args.trainSet}_{args.save_model}'),history.history)

