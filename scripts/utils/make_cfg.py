
from pickle import dump
import json


SAMPLE_DICT = {
               
               'filters':[8,16,32,64,128],
               'kernel_size':[3,3,3,3,3],
               'conv_padding':['same','same','same','same','same'],
               'activation':['relu','relu','relu','relu','relu'],
               'pool_size':[2,2,2,2,2],
               'stride':[None,None,None,None,None],
               'pool_padding':['valid','valid','valid','valid','valid'],
               
}


def make_cfg_file(
         input_dict:dict[str,any] = SAMPLE_DICT,
         filename:str = f'config/default_config',
         extention:str = 'cfg',
         )->None:

    '''
    writes a configuration file for the backbone of the network from a dictionary, a example of
    how to write this dictionary is provided on sample dictionary within this file
    '''

    
    if extention == 'cfg':
        with open(f'{filename}.{extention}','wb') as conf:
            dump(input_dict,conf)
            conf.close()
    
    if extention == 'json':
        with open(f'{filename}.{extention}','w') as conf:
            json.dump(input_dict,conf,indent = 6)
            conf.close()
    


if __name__ == '__main__' :
    make_cfg_file()
    



