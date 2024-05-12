import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
import yaml

import numpy as np
from random import sample
from pickle import dump
import json

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

def train_batch_consolidation(train_x_inputs:list[dict[str,str]],x_samples:list[any],tran_y_inputs:list[dict[str,str]],y_samples:list[any],max_boxes:int,IMG_SHAPE:tuple[int],num_clases:int)->list[np.array,dict]:
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
        potential_labels = [x for x in range(num_clases)]
        pl = potential_labels.copy()
        image_processed = process_image(individual_image,IMG_SHAPE)
        x_samples.append(image_processed)
        image_annotations = [annotation for annotation in tran_y_inputs if annotation['image_id'] == individual_image['id']]

        # width_inverse, inverse_height = 1 / individual_image['width'] , 1 / individual_image['height'] # No es necesario pues las bbox ahora se normalizan desde el archivo COCO
        # bbox_normalize_matrix = np.array([[width_inverse , 0, 0, 0],[0,inverse_height,0,0],[0,0,width_inverse,0],[0,0,0,inverse_height]],dtype=np.float32)
        # dict_load(total_targets_per_image,individual_image['file_name'],f'image') anexo nombre del archivo a las variables de anotaciones y cajas

        for annotation in image_annotations:
            label_vec = np.zeros(len(pl))
            ci = annotation['category_id']
            label_vec[int(ci)] = 1
            # ann_per_image.append(ci)
            if ci in potential_labels:
                potential_labels.remove(ci)

            # new_value_box = tuple(np.dot(np.array(annotation["bbox"],dtype=np.float32),bbox_normalize_matrix)) 

            dict_load(total_targets_per_image,np.array(annotation["bbox"],dtype=np.float32),f'bbox_head{int(ci)}')
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

def read_data(coco_file:str)->list[list[dict[str,str]],list[dict[str,str]],list[dict[str,str]],int]:
    '''
    json read and extration of information into lists:
    
    outputs  images : list of image atributes including filepath, size, ...
             annotations : list of annotations of all images along with bbox location, segmentation values ....
             categories : list of categories and their respective encoding
             num_classes : int of number of categories 
    '''
    
    with open(coco_file, 'r') as f:
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


def txt_file_information_colector(filename:str)->dict:
    '''
    function dedicated to read the txt files that contains the annotations per images 
    (ground/prediction) an return it as a dict
    '''
    image_annotations = {
        'categories' : [],
        'v1' : [],
        'v2' : [],
        'v3' : [],
        'v4' : [],
    }

    with open(filename) as file:
        lines = file.readlines()
        
        for line in lines:
            value_list = line.split(' ')
        
            for key,value in zip(image_annotations.keys(),value_list):
                image_annotations[key].append(value.replace('\n',''))

    return image_annotations


def diff_annotations(df1_o:dict,df2_o:dict)->float:

    '''
    calculates how equal de predictions and ground trues are by reading dicts with the information and comparing 
    number of prediction and what those prediction actually are, by subtracting the predicted quantities and the 
    ground ones, returns a float between 0 - 1 being 0 a bad score (not similar at all) and 1 (identical)
    '''

    SCORE = 0
    df1 = df1_o.copy()
    df2 = df2_o.copy()

    diff_of_lens = abs(len(df1['categories']) - len(df2['categories']))

    if diff_of_lens == 0:
        SCORE +=1


    ctg1 = set(df1['categories']) #.categories.astype(str).unique()
    ctg2 = set(df2['categories']) #.categories.astype(str).unique()
    

    if len(ctg1) == len(ctg2):
        ctg_shortest = ctg1
        ctg_longest = ctg2
    else:
        ctg_shortest = ctg1 if len(ctg1) < len(ctg2) else ctg2 #get_length_assert(ctg1,ctg_min) if get_length_assert(ctg1,ctg_min) else ctg2
        ctg_longest = ctg1 if len(ctg1) > len(ctg2) else ctg2 #ctg1 if ctg1.sort() != ctg_shortest.sort() else ctg2
    
    similarity_unit = 1 / len(ctg_longest)
    # similarity_score_categories = 0
    
    for ctg in ctg_shortest : 
        if ctg in ctg_longest:
            SCORE += similarity_unit

    del df1['categories'] #.drop('categories',axis = 1)
    del df2['categories'] #.drop('categories',axis = 1)

    pos_diff = []
    for value1,value2 in zip(df1.values(),df2.values()):
        for i,j in zip(value1,value2):
            i,j = float(i),float(j)
            pos_diff.append(abs(i-j))

    df_diff_magnitud = sum(pos_diff)
    max_diff_magnitud = len(pos_diff)
    
    diff_magnitud_normalize = df_diff_magnitud / max_diff_magnitud
    SCORE += (1 - diff_magnitud_normalize)
    # # print(diff_magnitud_normalize,similarity_unit,diff_of_lens,SCORE)

    return SCORE/3

def bbox_COCO_format(bbox:list[float])->list[float]:
    bbox = [float(x) for x in bbox]
    x_center = 0.5 * (bbox[0] + bbox[2])
    y_center = 0.5 * (bbox[1] + bbox[3])
    width = np.abs(bbox[2] - bbox[0])
    height = np.abs(bbox[3] - bbox[1])
    return [x_center,y_center,width,height]

def bbox_de_COCO_format(bbox:list[float])->list[float]:
    bbox = [float(x) for x in bbox]
    x_min = bbox[0] - (0.5 * bbox[2])
    y_min = bbox[1] - (0.5 * bbox[3])
    x_max = x_min + bbox[2]
    y_max = y_min + bbox[3]
    return [x_min,y_min,x_max,y_max]


def saveModel(model:tf.Tensor,save_path:str,results:dict[str,any] = None)->None:

    model.save(save_path)#os.path.join(save_path, args.save_model))
    
    if results :
        with open(f'{save_path[:-4]}_results.json', 'w') as f:
            json.dump(results, f)

    print(f"Model saved to {save_path}")