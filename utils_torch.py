import torch
import torchvision.transforms as transforms
import numpy as np
import json
import os
from random import sample
from pickle import dump
import yaml
from tqdm import tqdm
from PIL import Image
from typing import List,Dict,Callable,Iterable,Tuple

def process_image(image_dict:Dict[str,any], IMG_SHAPE:Tuple[int,int])->Iterable[float]:
    image_file_name = image_dict['file_name']
    # image_size = (image_dict['height'], image_dict['width'])

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SHAPE),
        transforms.ToTensor(),
    ])

    og_image = Image.open(image_file_name).convert("RGB")
    resized_image = transform(np.array(og_image))

    return resized_image.numpy()

def dict_load(mem:Dict[any,any], new_entry:any, key:any)->Dict[any,any]:
    try:
        mem[key].append(new_entry)
    except KeyError:
        mem[key] = [new_entry]
    return mem

def image_train_set(total_images:List[str], sample_size:int)->List[str]:
    return sample(total_images, int(sample_size * len(total_images)))

def image_test_set(total_image_set:List[str], train_image_set:List[str])->List[str]:
    return [image for image in total_image_set if image not in train_image_set]

def get_maximum_number_of_annotation_in_set(annotations:Dict[str,any], images:Dict[str,any])->int:
    annotation_id_per_image = [[x['category_id'] for x in annotations if x['image_id'] == y['id']] for y in images]
    annotation_count_by_class_id = [[annotation.count(x) for x in annotation] for annotation in annotation_id_per_image]
    max_count_annotation_id_per_image = [max(x) for x in annotation_count_by_class_id]
    max_boxes_per_id = max(max_count_annotation_id_per_image)
    return max_boxes_per_id

def train_batch_consolidation(train_x_inputs:Iterable[Dict[str,any]], x_samples:List[Iterable], train_y_inputs:Iterable[Dict[str,any]], y_samples:List[Dict[str,Iterable]], max_boxes:int, IMG_SHAPE:Tuple[int,int])->List[Iterable]:
    
    for individual_image in train_x_inputs:
        total_targets_per_image = {}

        potential_labels = [x for x in range(6)]
        pl = potential_labels.copy()

        image_processed = process_image(individual_image, IMG_SHAPE)
        x_samples.append(image_processed)
        
        image_annotations = [annotation for annotation in train_y_inputs if annotation['image_id'] == individual_image['id']]

        width_inverse, inverse_height = 1 / individual_image['width'], 1 / individual_image['height']
        bbox_normalize_matrix = np.array([[width_inverse, 0, 0, 0], [0, inverse_height, 0, 0], [0, 0, width_inverse, 0], [0, 0, 0, inverse_height]], dtype=np.float32)

        for annotation in image_annotations:
            label_vec = np.zeros(len(pl))
            ci = annotation['category_id']
            label_vec[int(ci)] = 1


            if ci in potential_labels:
                potential_labels.remove(ci)

            new_value_box = tuple(np.dot(np.array(annotation["bbox"], dtype=np.float32), bbox_normalize_matrix))
            
            # print(new_value_box)

            dict_load(total_targets_per_image, new_value_box, f'bbox_head{int(ci)}')
            dict_load(total_targets_per_image, label_vec, f'class_head{int(ci)}')
            total_targets_per_image[f'class_head{int(ci)}'] = [x for x in set(tuple(i) for i in total_targets_per_image[f'class_head{int(ci)}'])]

        for p in potential_labels:
            non_existing_on_image = np.zeros(len(pl))
            non_existing_on_image[int(p)] = 1
            dict_load(total_targets_per_image, tuple(np.array([0, 0, 0, 0], dtype=np.float32)), f'bbox_head{int(p)}')
            dict_load(total_targets_per_image, non_existing_on_image, f'class_head{int(p)}')

        for key in total_targets_per_image.keys():
            if 'bbox_head' in key:
                while len(total_targets_per_image[key]) < max_boxes:
                    total_targets_per_image[key].append(tuple(np.array([0, 0, 0, 0], dtype=np.float32)))

        y_samples.append(total_targets_per_image)


    y_targets = {
        key: np.stack([x[key] for x in y_samples], axis=0) for key in y_samples[0].keys()
    }

    x_targets = np.stack(x_samples, axis=0)

    return [x_targets, y_targets]

def pytorchDataConsolidation(targets:Dict[str,Iterable],images:Iterable):
    targets['images'] = images

    training_data_list = []
    for i in tqdm(range(len(targets['images'])), total=len(targets['images'])):
        row = []
        for value in targets.values():
            row.append(value[i])
        training_data_list.append(row)
    return training_data_list

def read_data():
    with open(os.path.join(os.getcwd(), 'annotations', 'dataset.json'), 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    num_classes = len(categories)
    return images, annotations, categories, num_classes

def dump_file(file, filename):
    with open(filename, 'wb') as doc:
        dump(file, doc)

def display_yaml_content(yaml_file, return_data=True):
    with open(yaml_file) as file:
        documents = yaml.full_load(file)
        for item, doc in documents.items():
            print(item, ':', doc)
    return documents if return_data else None

def config_yaml(train_path, validation_path, categories, config_file_name='default'):
    config_dict = {
        'train': train_path,
        'val': validation_path,
        'names': categories
    }

    with open(f'{config_file_name}.yaml', 'w') as config_file:
        yaml.dump(config_dict, config_file, default_flow_style=False)

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
