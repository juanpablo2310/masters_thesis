#!/usr/bin/env python
# coding: utf-8
 
from dataclasses import dataclass, field
from typing import Dict, List 
import functools

import os
import datetime as dt
import pandas as pd
import numpy as np
from paths import get_project_labels,get_project_data_UN_dir,get_project_data_MELU_dir,get_project_annotations
from cv2 import imread, imshow, rectangle,putText,FONT_HERSHEY_SIMPLEX
import argparse
# from google.colab.patches import cv2_imshow
parser = argparse.ArgumentParser()
parser.add_argument('--dataSet',type = str,choices=['UN','MELU'], default = 'UN')
parser.add_argument('--labelFile',type = str, default = 'etiquetas.csv')
parser.add_argument('--fileName',type = str, default = 'dataset.json')
args = parser.parse_args()

@dataclass 
class JsonCOCOFormatter:
    
    ##### creacion de formato
    info: Dict = field(init = False, default_factory =  dict),
    licenses: List[Dict] = field(default_factory = list),
    images: List[Dict] = field(default_factory = list),
    annotations: List[Dict] = field(default_factory = list),
    categories: List[Dict] = field(default_factory = list), 
    # segment_info: List[Dict] = field(default_factory = list), 
    
    ##### input variables
    url : str = field(default="None")
    id_lincense : int = field(default = None)

    lincense_name : str = field(default = "generic_lincense")
    file_name : str = field(default = "")
    coco_url : str = field(default = "None")
    height : int = field(default = None)
    width : int = field(default = None)
    date_captured : str = field(default = "None")
    flickr_url : str = field(default = "None")
    id_photo : int = field(default = None)
 
    category_name : str = field(default = "")
    subcategory_name : str = field(default = "")
    id_category : int = field(default = None)
 
    segmentation : List[int] = field(default_factory = list)
    is_crowd : str = field(default = "FALSE")
    bbox : List[int] = field(default_factory = list)
    id_annotation : int = field(default = None)
    SAVE_PATH : str = field(default = get_project_annotations())
    FILE_NAME_s : str = field(default = '')
   

    def __post_init__(self):
        self.info = {
                    "year": dt.datetime.now().year, #"2023",
                    "version": "1.0",
                    "description": "National Herbarium DataSet",
                    "contributor": "DataLab Universidad Nacional de Colombia",
                    "url": "",
                    "date_created": f"{dt.datetime.now().year}/{dt.datetime.now().month}/{dt.datetime.now().day}"
                    }
    
    def make_lincense(self):
        return self.licenses.append({
                "url": self.url,
                "id": self.id_lincense,
                "name": self.lincense_name
            })
        
    def make_image(self):
        condition = True
        for x in self.images:
            if self.file_name in x.values():
                condition = False
        
        if condition:
            return self.images.append({
                "license": self.id_lincense,
                "file_name": self.file_name,
                "coco_url": self.coco_url,
                "height": self.height,
                "width": self.width,
                "date_captured": self.date_captured,
                "flickr_url": self.flickr_url,
                "id": self.id_photo,
                # "annotations" : self.annotations
                })
        
    def make_category(self):
        condition = True
        for x in self.categories:
            if self.subcategory_name in x.values():
                condition = False
        if condition:
            return self.categories.append({
                    "supercategory": self.category_name,
                    "id": self.id_category,
                    "name": self.subcategory_name
                })
            
    def make_annotation(self):
        return self.annotations.append({
                "segmentation": self.segmentation,
                "area": self.height * self.width,
                "iscrowd": self.is_crowd,
                "image_id": self.id_photo,
                "bbox": self.bbox,
                "category_id": self.id_category,
                "id": self.id_annotation
            })
            
    def to_file(self):
        target_keys =['info','licenses','images','annotations','categories']
        target_dict = {key: value for key, value in self.__dict__.items() if key in target_keys}
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
        with open(os.path.join(self.SAVE_PATH,self.FILE_NAME_s),'w+') as file:
            # file.write(str(target_dict).replace('"',"'").strip('"<>()'))
            file.write(str(target_dict).replace("'",'"').strip("'<>()"))
    

def assign_number_according_to_class(series : pd.Series):
    hash_dict = {}
    for i,label in enumerate(series.unique()):
        hash_dict[label] = i
    return hash_dict

def put_hash_column(table : pd.DataFrame):
    hash_dict = assign_number_according_to_class(table['class_name'])
    for k,v in hash_dict.items():
        table.loc[table.class_name == k,'class_hash'] = int(v)
    return table

def bbox_COCO_format(bbox:list[float])->list[float]:
    x_center = 0.5 * (bbox[0] + bbox[2])
    y_center = 0.5 * (bbox[1] + bbox[3])
    width = np.abs(bbox[2] - bbox[0])
    height = np.abs(bbox[3] - bbox[1])
    return [x_center,y_center,width,height]

def bbox_de_COCO_format(bbox:list[float])->list[float]:
    x_min = bbox[0] - (0.5 * bbox[2])
    y_min = bbox[1] - (0.5 * bbox[3])
    x_max = x_min + bbox[2]
    y_max = y_min + bbox[3]
    return [x_min,y_min,x_max,y_max]

def draw_bbox(image_path:str,ids:list[int],coordinates:list[list[float]],cloud:bool=False):
    image_to_visualize = image_path #os.path.join(IMG_MEL_PATH,image_max_name)
    img = imread(image_to_visualize)
    for id,coordinate in zip(ids,coordinates) :
        nc = bbox_de_COCO_format([float(x.replace('\n','')) for x in coordinate])
        rectangle(img,(int(nc[0]*img.shape[1]),int(nc[1]*img.shape[0])),((int(nc[2]*img.shape[1]),int(nc[3]*img.shape[0]))),(255,0,0),3)
        putText(img, id, (int(nc[0]*img.shape[1])-20,int(nc[1]*img.shape[0])+10), FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    if cloud : cv2_imshow(img)
    else : imshow(img) 

def main(data_table:pd.DataFrame,fileName = args.fileName,cocoFormat = True)->None:

    if 'valid_path' in data_table.columns:
        data_table = data_table.drop(columns = 'valid_path')

    data_table = put_hash_column(data_table)


    file_information_gather = data_table.groupby('filename')
    id_picture = 0
    id_licencia = 0
    imagenes , licensias, anotaciones, categorias = [],[],[],[]
    for picture_name, sub_table in file_information_gather:
        id_picture += 1
        id_licencia += 1
        id_anotacion = 0
        # picture_path = os.path.join(os.getcwd(),'drive','MyDrive','imagenes', picture_name)#os.path.join(os.getcwd(),'data','imagenes', picture_name)
        picture_path = get_project_data_UN_dir(f'imagenes/{picture_name}') if args.dataSet == 'UN' else get_project_data_MELU_dir(f'train/images/{picture_name}') #os.path.abspath(os.path.join('data','imagenes',picture_name))
        single_photo_data = functools.partial(JsonCOCOFormatter,images=imagenes,licenses=licensias,annotations=anotaciones,categories=categorias,file_name=picture_path,height=sub_table.height.unique()[0],width=sub_table.width.unique()[0],id_lincense = id_licencia, id_photo = id_picture,FILE_NAME_s = fileName)
        for i,row in sub_table.iterrows():
            id_anotacion += 1

            if cocoFormat:
                bbox_paste = bbox_COCO_format([row["xmin"]/row['width'],row["ymin"]/row['height'],row['xmax']/row['width'],row['ymax']/row['height']]) # coordenadas para dar puntos medios y tamaños de las cajas[(row["xmin"] + (0.5 * class_width))/row["width"],(row["ymin"] + (0.5 * class_height))/row["height"],class_width/row["width"],class_height/row["height"]]#
            else :
                bbox_paste = [row["xmin"],row["ymin"],row['xmax'],row['ymax']]

            whole_data = single_photo_data(category_name = "etiquetas", subcategory_name = row['class_name'],segmentation = ["empty"],bbox = bbox_paste,id_annotation=id_anotacion,id_category=row['class_hash'])
            whole_data.make_lincense()
            whole_data.make_category()
            whole_data.make_annotation()
            whole_data.make_image()
            

    whole_data.to_file()


if __name__ == '__main__':
    data_table = pd.read_csv(get_project_labels(args.labelFile))
    main(data_table)
