import os

def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

def get_project_data_UN_dir(imageName:str = '') -> str:
    return os.path.join(get_project_root(), 'data_UN',imageName)

def get_project_data_MELU_dir(imageName:str = '') -> str:
    return os.path.join(get_project_root(), 'data_MELU',imageName)

def get_project_annotations(annotationFile:str = '') -> str:
    return os.path.join(get_project_root(), 'annotations',annotationFile)

def get_project_models(model:str = '') -> str:
    return os.path.join(get_project_root(), 'models', model)

def get_project_configs(confFile:str = '') -> str:
    return os.path.join(get_project_root(), 'config', confFile)

def get_project_results(fileResult:str = '') -> str:
    return os.path.join(get_project_root(), 'results_analysis',fileResult)

def get_project_scripts(scriptName:str = '') -> str:
    return os.path.join(get_project_root(), 'scripts',scriptName)
 
def get_project_labels(databaseName:str = '') -> str:
    return os.path.join(get_project_root(), 'labels_databases',databaseName)
