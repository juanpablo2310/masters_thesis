import tensorflow as tf

import numpy as np
import argparse
from pickle import load

from utils import get_maximun_number_of_annotation_in_set, train_batch_consolidation,image_train_set,image_test_set,read_data,dump_file
from models import model_inputs,model_backbone,model_outputs,model_consolidation


parser = argparse.ArgumentParser(description='trains the custom architecture')
parser.add_argument('--epochs',type = int, default=1)
parser.add_argument('--batch_size',type = int, default = 1)
parser.add_argument('--model_structure',type = str, default = 'config/default_config.cfg')
parser.add_argument('--save_model',type = str,default = None)
parser.add_argument('--device',type=str,default='CPU')
args = parser.parse_args()


IMG_SHAPE = (500, 500)

config_object = open(args.model_structure,'rb')
network_structure = load(config_object)

images,annotations,_,num_classes = read_data()

train_images = image_train_set(images,0.75)
dump_file(train_images,'sets/train_image_set')
test_images = image_test_set(images,train_images)
dump_file(test_images,'sets/test_image_set')

total_training_images = [] # 'x'
total_training_targets = [] #'y'

max_n_boxes = get_maximun_number_of_annotation_in_set(annotations,images)  # calculo el numero maximo de cajas por categoria sobre el set total de muestras para garantizar que todas las instancias van a ser detectadas

images_for_training , training_targets = train_batch_consolidation(
                                                                train_images,
                                                                total_training_images,
                                                                annotations,
                                                                total_training_targets,
                                                                max_n_boxes,
                                                                IMG_SHAPE
                                                                )

normalization_value = np.max([x[0][0] for x in images_for_training])


layer_inputs = model_inputs(IMG_SHAPE)
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
                        verbose=1
                        )

if args.save_model:
    base_model.save(f'models/CNN_model{args.save_model}')
