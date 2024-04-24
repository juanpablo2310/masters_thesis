import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input,Reshape
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.metrics import IoU,MeanIoU,AUC,Precision,Recall,Accuracy
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
import pdb
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))))

try:
    from utils_tensorflow import dict_load
except ModuleNotFoundError:
    from scripts.tensorflowLocal.utils_tensorflow import dict_load

class CustomIoUMetric(tf.keras.metrics.Metric):

    '''
    clase que permite la evaluacion de IoU para valores negativos en la matriz de confusion resultante de usar funciones de activacion como softmax, linear o sigmoide en la 
    regresion para las cajas
    '''
    def __init__(self, name='custom_iou', **kwargs):
        super(CustomIoUMetric, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true:list[int], y_pred:list[int], sample_weight=None)->None:
        
        y_true = tf.clip_by_value(y_true, 0, 1)  # Ensure labels are in [0, 1]
        y_pred = tf.clip_by_value(y_pred, 0, 1)
        intersection = tf.reduce_sum(tf.minimum(y_true, y_pred))
        union = tf.reduce_sum(tf.maximum(y_true, y_pred))
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self)->tf.Tensor:
        return self.intersection / (self.union + tf.keras.backend.epsilon())

    def reset_state(self):
        
        self.intersection.assign(tf.zeros_like(self.intersection))
        self.union.assign(tf.zeros_like(self.union))
        
    
    def get_config(self):
        config = super(CustomIoUMetric, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def  model_inputs(size:tuple=(None,None))->tf.keras.layers:
    '''
    reads the desired size for input images and returns a tf input layer with the inputed size
    inputs:  size type tuple
    outputs : input_layer type keras.layer
    '''

    input_shape = (size[0],size[1],3,)
    input_layer = Input(input_shape)
    return input_layer


def model_backbone(input_layer:tf.keras.layers,params:dict[str,any],normalization_value:float)->tf.keras.layers:
    '''
    compiles the classic structure of convnet plus maxpooling for especific parameters inputed in a dictionary, this 
    structure replicates as many times as parameters where inputed, the number of parameters should be the same for all
    of them
    inputs : input_layer: type keras.layer with the specify shape of the input images
             paramas: type dict with all the parameters need it, an example is given in DEFAULT_PARAMS which is the default input for this function
             normalization_value : maximun value of the pixels after load the image as an array to normalize all the values

    outputs : backbone: keras.layers containg all the feature extraction architecture
    '''

    backbone = Rescaling(1./normalization_value, name='bl_1')(input_layer)

    block_numbers = set([len(x) for x in params.values()])

    if (len(block_numbers) > 1) | (len(block_numbers) == 0):
        raise ValueError('arrays empty or with different length')
 
    for i in range(list(block_numbers)[0]):
        backbone = Conv2D(filters = params['filters'][i], kernel_size = params['kernel_size'][i], padding = params['conv_padding'][i], activation= params['activation'][i], name=f'bl_conv_{i}')(backbone)
        backbone = MaxPooling2D(pool_size = params['pool_size'][i], strides = params['stride'][i],padding = params['pool_padding'][i], name=f'bl_maxpol_{i}')(backbone)
    backbone = Flatten(name='bl_flatten')(backbone)
    return backbone


def model_outputs(backbone:tf.keras.layers,num_classes:int,max_n_boxes:int)->tf.keras.layers:
    '''
    is a double ended layer that gathers on one side the predictions for the bbox and in the other a horizontal vector 
    containig the probability distribution for the label of that specific set of bbox
    inputs : backbone : type keras.layer feature extraction results
             num_classes : int containing the number of classes to identify
             max_n_boxes : int with the maximun number of boxes for a single class accross all images
    
    outputs : bbox_outputs : set of size (max_n_boxes, 4) with coordinates for the boxes locations
              class_outputs : set of size (1, num_classes) with the probability distribution for a particular set of bbox
    '''

    bbox_outputs = []
    class_outputs = []
    for i in range(num_classes):
        bbox_head = Dense(128, activation='relu')(backbone)
        bbox_head = Dense(max_n_boxes * 4, activation='relu')(bbox_head)
        bbox_head = Reshape((max_n_boxes, 4), name=f'bbox_head{i}')(bbox_head)
        bbox_outputs.append(bbox_head)
        
        label_classifier = Dense(128, activation='relu', name=f'cl_{i}_1')(backbone)
        label_classifier = Dense(num_classes, activation='softmax', name=f'cl_{i}_2')(label_classifier) # Binary classification for each class
        label_classifier = Reshape((1, num_classes), name=f'class_head{i}')(label_classifier)
        class_outputs.append(label_classifier)
    return bbox_outputs,class_outputs


def model_consolidation(num_classes:int,input_layer:tf.keras.layers,bbox_outputs:list[tf.keras.layers],class_outputs:list[tf.keras.layers],show:bool=True)->tf.keras.models:
    '''
    compilation of all layer into one single model, definition of cost functions for each output channel
    inputs : num_classes : int of all number of classes to be detected
             input_layer : keras.layer
             bbox_outputs : keras.layer
             class_outputs : keras.layer
    '''
    
    losses = {}
    for i in range(num_classes):
        dict_load(losses,CategoricalCrossentropy(num_classes),f'class_head{i}')
        dict_load(losses,MeanSquaredError(),f'bbox_head{i}')

    base_conv_model = Model(input_layer, outputs=bbox_outputs + class_outputs)
    base_conv_model.compile(loss=losses, optimizer=Adam(), metrics=[CustomIoUMetric(), AUC(),Accuracy(),Precision(),Recall()]) #num_classes,'accuracy',MeanIoU(num_classes),
    if show:
        print(base_conv_model.summary())
    return base_conv_model
