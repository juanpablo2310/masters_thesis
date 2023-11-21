
from pickle import dump


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
         filename:str = 'default_config.cfg'
        )->None:

    '''
    writes a configuration file for the backbone of the network from a dictionary, a example of
    how to write this dictionary is provided on sample dictionary within this file
    '''

    with open(filename,'wb') as conf:
        dump(input_dict,conf)
        conf.close()


if __name__ == '__main__' :
    make_cfg_file()
    



