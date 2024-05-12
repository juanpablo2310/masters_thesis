
import numpy as np
import json
import os
from random import sample
from pickle import dump
import yaml
from tqdm import tqdm
from cv2 import imread
from typing import List,Dict,Callable,Iterable,Tuple,Union,Any
from torchvision.datasets.coco import CocoDetection

import torch

import pdb


class CustomCocoDetection(CocoDetection):
    
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile, transform=transform)

    def _load_target(self, id: int) -> List[Any]:
            return self.coco.imgToAnns[id]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
            image, target = super().__getitem__(index)
            if self.transforms is not None:
                image, target = self.transforms(image, target)
            return image, target



def normalize_image(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  
    return matrix
 

def custom_collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        img_normalize = normalize_image(np.array(img))
        images.append(torch.from_numpy(img_normalize)) 
        targets.append(target)
    images = torch.stack(images, dim=0)
    return images, targets

def calculate_mean_std_per_channel(image_folder:str)->List[float]:
    
    all_means = []
    all_stds = []

    for file in os.listdir(image_folder):
        if file.endswith('.png'):
            image = imread(os.path.join(image_folder,file))
            means_by_channels = np.mean(image,axis=(0,1))
            stds_by_channels = np.std(image,axis=(0,1))
            all_means.append(means_by_channels)
            all_stds.append(stds_by_channels)

    arrange_means = np.stack(all_means)
    arrange_stds = np.stack(all_stds)

    return np.mean(arrange_means,axis=0),np.mean(arrange_stds,axis=0)

def process_image(image_dict:Dict[str,any], transform:Callable = None)->Iterable[float]:
    image_file_name = image_dict['file_name']
    og_image = imread(image_file_name) #Image.open(image_file_name).convert("RGB")
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

def pytorchDataConsolidation(targets:Dict[str,Iterable],images:Iterable):
    targets['images'] = images

    training_data_list = []
    for i in tqdm(range(len(targets['images'])), total=len(targets['images'])):
        row = []
        for value in targets.values():
            row.append(value[i])
        training_data_list.append(row)
    return training_data_list

def open_json_file(file_path:str)->Dict[any,any]:
    coco_file = open(file_path, 'r')
    return json.load(coco_file)


def read_data(coco_file)->List:
    # default_path = os.path.join(os.getcwd(), 'annotations', 'dataset.json')
    coco_data = open_json_file(coco_file)
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    num_classes = len(categories)
    return images, annotations, categories, num_classes

def dump_file(file:str, filename:str)->None:
    with open(filename, 'wb') as doc:
        dump(file, doc)

def display_yaml_content(yaml_file:str, return_data:bool=True)->Union[Dict,None]:
    with open(yaml_file) as file:
        documents = yaml.full_load(file)
        for item, doc in documents.items():
            print(item, ':', doc)
    return documents if return_data else None

def config_yaml(train_path:str, validation_path:str, categories:Dict[int,str], config_file_name:str='default')->None:
    config_dict = {
        'train': train_path,
        'val': validation_path,
        'names': categories
    }

    with open(f'{config_file_name}.yaml', 'w') as config_file:
        yaml.dump(config_dict, config_file, default_flow_style=False)

def txt_file_information_colector(filename:str)->Dict[str,Union[int,float]]:
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

def saveModel(model:torch.nn.Module,save_path:str,results:Dict[str,any] = None)->None:

    torch.save(model.state_dict(), save_path) #os.path.join(save_path, args.save_model))
    
    if results :
        with open(f'{save_path[:-4]}_results.json', 'w') as f:
            json.dump(results, f)

    print(f"Model saved to {save_path}")
