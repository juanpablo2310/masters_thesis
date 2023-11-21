import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
import yaml

import numpy as np
from random import sample
import json 
import os 
from pickle import dump


def process_image(image_dict:dict[str,str],IMG_SHAPE:tuple[int]) -> np.array:
    ''' 
    this fuction recives a dictionary of images locations and atributes, loads the actual file, reshape the image to 500x500
    and normalize de values of the image so the go from 0 to 1 and cast each element of the array as float32

    inputs : image_dict (type dict)
    outputs : image_array (type numpy.array)
    '''

   

    image_file_name = image_dict['file_name']
    image_size = (image_dict['height'], image_dict['width'])
    og_image = load_img(image_file_name, target_size = image_size)
    imaga_re_size = tf.image.resize(og_image,IMG_SHAPE) #,preserve_aspect_ratio = True, antialias=True
    image_array = img_to_array(imaga_re_size)
    # image_array = tf.cast(image_array, tf.float32) / np.mean(IMG_SHAPE)

    return image_array
    
def dict_load(mem:dict[str,any],new_entry:any,key:str):
    '''
    this function populate a dictionary adding the value under the specify key and adds the key en case isn't there yet
    
    inputs : mem (type dict)
             new_entry (type any)
             key (type str)

    outputs : mem (type dict)
    '''
    try:
        mem[key].append(new_entry)
    except KeyError:
        mem[key] = [new_entry]
    return mem

def image_train_set(total_images:list[any],sample_size:float)->list[any]:
    '''
    this function takes a list and a number between 0 and 1 to returns a shorter sampled list scaled by the number provided

    inputs : total_images (type list) 
             sample_size (type float)

    '''
    
    return sample(
        total_images, int(sample_size*len(total_images)
                                   ))

def image_test_set(total_image_set:list[any],train_image_set:list[any])->list[any]:

    '''
    takes two list and returns the elements that are present in bigger list that are not in the shorter one

    inputs : total_image_set (type list)
             train_image_set (type list)
    '''
    
    return[
        image  for image in total_image_set if image not in train_image_set
    ]

def get_maximun_number_of_annotation_in_set(annotations:list[dict[str,str]],images:list[dict[str,str]])->int:
    '''
    this function takes 2 dicts and construct sublist base on id relationship between its elements and then takes the maximun length
    of the sublist 

    inputs annotations (type list)
           images (type list)

    outputs max_boxes (type int)
    '''
    annotation_id_per_image = [[x['category_id'] for x in annotations if x['image_id'] == y['id']] for y in images]
    annotation_count_by_class_id = [[annotation.count(x) for x in annotation] for annotation in annotation_id_per_image]
    max_count_annotation_id_per_image = [max(x) for x in annotation_count_by_class_id] #annotations_lens = [len(x) for x in annotation_count_by_class]
    max_boxes_per_id = max(max_count_annotation_id_per_image)
    return max_boxes_per_id

def train_batch_consolidation(train_x_inputs:list[dict[str,str]],x_samples:list[any],tran_y_inputs:list[dict[str,str]],y_samples:list[any],max_boxes:int,IMG_SHAPE:tuple[int])->list[np.array,dict]:
    '''
    This function takes the list of features and parameters for the neural network and compiles them into suitable stack of dicts
    and tensors that are consume by the model

    inputs  train_x_inputs (type list of images paths and properties)
            x_samples (type empty list )
            train_y_inputs (type list annotations for each image of the set)
            y_samples (type empty list)
            max_boxes (type int maximun number of detection of the same class accross all images)
    
    outputs x_targets (type np.array of np.arrays representing images for set)
            y_targets (type dict of class id and bbox location grouped by each image)
    '''
    
    for individual_image in train_x_inputs:
        # ann_per_image = []
        total_targets_per_image = {}
        potential_labels = [x for x in range(6)]
        pl = potential_labels.copy()
        image_processed = process_image(individual_image,IMG_SHAPE)
        x_samples.append(image_processed)
        image_annotations = [annotation for annotation in tran_y_inputs if annotation['image_id'] == individual_image['id']]

        width_inverse, inverse_height = 1 / individual_image['width'] , 1 / individual_image['height']
        bbox_normalize_matrix = np.array([[width_inverse , 0, 0, 0],[0,inverse_height,0,0],[0,0,width_inverse,0],[0,0,0,inverse_height]],dtype=np.float32)
        # dict_load(total_targets_per_image,individual_image['file_name'],f'image') anexo nombre del archivo a las variables de anotaciones y cajas

        for annotation in image_annotations:
            label_vec = np.zeros(len(pl))
            ci = annotation['category_id']
            label_vec[int(ci)] = 1
            # ann_per_image.append(ci)
            if ci in potential_labels:
                potential_labels.remove(ci)
            new_value_box = tuple(np.dot(np.array(annotation["bbox"],dtype=np.float32),bbox_normalize_matrix)) 
            dict_load(total_targets_per_image,new_value_box,f'bbox_head{int(ci)}')
            dict_load(total_targets_per_image,label_vec,f'class_head{int(ci)}')
            total_targets_per_image[f'class_head{int(ci)}'] = [x for x in set(tuple(i) for i in total_targets_per_image[f'class_head{int(ci)}'])]  # cuando teniamos 0 o 1 solamente como opciones quitamos los repetidos con set
                                                               
        for p in  potential_labels:
            non_existing_on_image = np.zeros(len(pl))
            non_existing_on_image[int(p)] = 1
            dict_load(total_targets_per_image,tuple(np.array([0,0,0,0],dtype=np.float32)),f'bbox_head{int(p)}')
            dict_load(total_targets_per_image,non_existing_on_image,f'class_head{int(p)}')
            
        
        for key in total_targets_per_image.keys():
            if 'bbox_head' in key : 
                while len(total_targets_per_image[key]) < max_boxes :
                    total_targets_per_image[key].append(tuple(np.array([0, 0, 0, 0],dtype=np.float32)))

            
       
        y_samples.append(total_targets_per_image)


    y_targets = {
        key:np.stack([x[key] for x in y_samples], axis=0) for key in y_samples[0].keys()
        }
    
    x_targets = np.stack(x_samples,axis=0)
    
    return [x_targets,y_targets]

def read_data()->list[list[dict[str,str]],list[dict[str,str]],list[dict[str,str]],int]:
    '''
    json read and extration of information into lists:
    
    outputs  images : list of image atributes including filepath, size, ...
             annotations : list of annotations of all images along with bbox location, segmentation values ....
             categories : list of categories and their respective encoding
             num_classes : int of number of categories 
    '''
    
    with open(os.path.join(os.getcwd(),'annotations','dataset.json'), 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    num_classes = len(categories)
    return images,annotations,categories,num_classes

def dump_file(file:any,filename:str)->None:
    '''
    uses pickle format to save any kind of python object to an external file, this are easy readeable back using method pickle.load
    '''
    with open(filename,'wb') as doc:
        dump(file,doc)
        doc.close()
    
def display_yaml_content(yaml_file:yaml,return_data:bool = True)->dict:
    '''
    displays and stores by default all information contain into yaml files required to train ultralitycs yolo version
    '''
    with open(yaml_file) as file:
        documents = yaml.full_load(file)
        for item,doc in documents.items():
            print(item,':',doc)
    return documents if return_data else None 

def confg_yaml(train_path:str,validation_path:str,categories:dict,config_file_name:str = 'default')->None:
    '''
    creates a configuration file for the ultralytics yolo package, recives the path to training
    and validation images along with a dictionary containing the desired target classes
    '''

    config_dict = {
        'train':train_path,
        'val': validation_path,
        'names' : categories
        }
    
    with open(f'{config_file_name}.yaml' , 'w') as config_file:
        yaml.dump(config_dict,config_file,default_flow_style=False)