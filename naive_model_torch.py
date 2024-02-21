import numpy as np
import argparse
from pickle import load
from tqdm import tqdm

import torch
from torch import nn,optim
from torchinfo import summary

from utils_torch import get_maximum_number_of_annotation_in_set,train_batch_consolidation,image_train_set,image_test_set,read_data,dump_file,pytorchDataConsolidation
from models_torch import ModelFromScratch,train,test

parser = argparse.ArgumentParser(description='trains the custom architecture')
parser.add_argument('--epochs',type = int, default=1)
parser.add_argument('--batch_size',type = int, default = 1)
parser.add_argument('--model_structure',type = str, default = 'config/torch_simple.cfg')
parser.add_argument('--save_model',type = str,default = None)
parser.add_argument('--device',type=str,default='CPU')
args = parser.parse_args()


IMG_SHAPE = (500, 500)

config_object = open(args.model_structure,'rb')
network_structure = load(config_object)

images,annotations,_,num_classes = read_data()

train_images = image_train_set(images,0.75)
dump_file(train_images,'sets/train_image_set')
test_images = image_test_set(images,train_images)
dump_file(test_images,'sets/test_image_set')

total_training_images = [] # 'x'
total_training_targets = [] #'y'

max_n_boxes = get_maximum_number_of_annotation_in_set(annotations,images) 

images_for_training , training_targets = train_batch_consolidation(
                                                                train_images,
                                                                total_training_images,
                                                                annotations,
                                                                total_training_targets,
                                                                max_n_boxes,
                                                                IMG_SHAPE
                                                                )

total_testing_images = []
total_testing_targests = []

images_for_testing , testing_targets = train_batch_consolidation(
                                                                test_images,
                                                                total_testing_images,
                                                                annotations,
                                                                total_testing_targests,
                                                                max_n_boxes,
                                                                IMG_SHAPE
                                                                )




training_data_list = pytorchDataConsolidation(training_targets,images_for_training)
testing_data_list = pytorchDataConsolidation(testing_targets,images_for_testing)

dataIterableTrain = torch.utils.data.DataLoader(training_data_list, shuffle=True)
dataIterableTest = torch.utils.data.DataLoader(testing_data_list, shuffle=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # # # # input_size = 784
# # # # hidden_size = 128
# # # # num_classes = 10
num_epochs = 10
learning_rate = 0.001
 

results = {}
  
# # # # Train and test the model with different activation functions
# # # for name, activation_function in activation_functions.items():
# # #     print(f"Training with {name} activation function...")
 
# # #     model = NeuralNetwork(input_size, hidden_size, num_classes, activation_function).to(device)
normalization_value = np.max([x[0][0] for x in images_for_training])
BasicModel = ModelFromScratch(network_structure,normalization_value,num_classes,max_n_boxes,IMG_SHAPE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(BasicModel.parameters(), lr=learning_rate)
 
train_loss_history = []
test_loss_history = []
test_accuracy_history = []
 
# # print('about to start training')

for epoch in tqdm(range(num_epochs), total=num_epochs):
    train_loss = train(BasicModel, dataIterableTrain, criterion, optimizer, device)
    test_loss, test_accuracy = test(BasicModel, dataIterableTest, criterion, device)

    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")

results = {
    'train_loss_history': train_loss_history,
    'test_loss_history': test_loss_history,
    'test_accuracy_history': test_accuracy_history
}

# # print(type(outputsFinal))