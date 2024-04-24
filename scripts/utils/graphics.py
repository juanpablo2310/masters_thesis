import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from utils.paths import get_project_results
from json import dump


def monochromaticIntensityHistogram(path : str,save:bool = False,show:bool = False):
    imageList = os.listdir(path)
    savePath = get_project_results(f'images/histograms/monochromatic/')
    os.makedirs(savePath,exist_ok=True)
    for image in imageList[:5]:
        im = cv2.imread(os.path.join(path,image))
        vals = im.mean(axis=2).flatten()
        counts, bins = np.histogram(vals, range(257))
        plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
        plt.xlim([-0.5, 255.5])
        plt.title('Histogram for intensity (grey scale) picture')
        plt.xlabel('Intensity value')
        plt.ylabel('Pixel Count')
        if save:
            plt.savefig(os.path.join(savePath,f'{image[:-4]}_histogram'),format = 'svg')
            with open(os.path.join(savePath,f'{image[:-4]}_histogram.json'),'w+') as file:
                dump({'bins':bins.tolist(),'counts':counts.tolist()},file,indent = 6)
        if show:
            plt.show()

def RGBhitogram(path:str, save:bool =False, show :bool = False):
    imageList = os.listdir(path)
    savePath = get_project_results(f'images/histograms/RGB/')
    os.makedirs(savePath,exist_ok=True)
    dict_results = {}
    for img_n in imageList[:5]:
        img = cv2.imread(os.path.join(path,img_n))
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv2.calcHist([img],[i],None,[256],[0,256])
            dict_results[col] = histr.tolist()
            plt.plot(histr,color = col)
            plt.xlim([0,256])
    
        plt.title('Histogram for color scale picture')
        plt.xlabel('Intensity value')
        plt.ylabel('Pixel Count')
        if save:
            plt.savefig(os.path.join(savePath,f'{img_n[:-4]}_histogram'),format = 'svg')
            with open(os.path.join(savePath,f'{img_n[:-4]}_histogram.json'),'w+') as file:
                dump(dict_results,file,indent = 6)
        if show:
            plt.show()

