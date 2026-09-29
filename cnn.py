# Image Classification Using CNNs and PyTorch: Identifying Document Types
# In this project, I developed a neural network model that classifies images into categories such as social security cards, driving licenses, and others. I used PyTorch to build the model, leveraging its unique feature of dynamic computational graphs, which sets it apart from other deep learning frameworks.


from google.colab import drive
drive.mount('/content/gdrive')

ROOT_DIR = "gdrive/My Drive/Datasets/CNN/"


#Importing Necessary files to read Images
import pandas as pd
import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline



# Printing random images from the dataset
Training_folder= ROOT_DIR + "Data/Training_data"
os.listdir(Training_folder)

from torchvision.datasets import ImageFolder
ImageFolder(Training_folder)

for path in os.listdir(Training_folder):
    for i in range(3):
        temp_path = Training_folder + "/" + path # Creating a temp path to work on images
        file = random.choice(os.listdir(temp_path)) # Randomly selecting an image from the temp pathb and saving them into a file
        image_path= os.path.join(temp_path, file) # Joining the file into the temp path folder and assigning it as the image path
        img = mpimg.imread(image_path)
        plt.figure(figsize=(5,5))
        plt.imshow(img)

#Setting the Image dimension and source folder for loading the dataset
IMG_WIDTH = 200
IMG_HEIGHT = 200
Train_folder = ROOT_DIR + 'Data/Training_data'
Test_folder = ROOT_DIR + 'Data/Testing_data'


#Creating the image data and the labels from the images
def create_dataset(Train_folder):
    img_data_array = []
    class_name = []
    classes = {'driving_license': [1,0,0], 'others': [0,1,0], 'social_security': [0,0,1]}
    for PATH in os.listdir(Train_folder):
        for file in os.listdir(os.path.join(Train_folder, PATH)):
            image_path= os.path.join(Train_folder, PATH,  file)
            image= cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float64')
            # image /= 255
            if len(image.shape) == 3:
                img_data_array.append(np.array(image).reshape([3, 200, 200]))
                class_name.append(classes[PATH])
    return img_data_array, class_name


# extract the image array and class name for training data
Train_img_data, train_class_name = create_dataset(Train_folder)

# extract the image array and class name for testing data
Test_img_data, test_class_name = create_dataset(Test_folder)
len(Train_img_data)


train_class_name[0]


# Implementing a CNN in PyTorch
# importing necessary libraries
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.utils.data as Data
from torch import Tensor
from torch.autograd import Variable


# version of pytorch
print(torch.__version__)

torch_dataset_train = Data.TensorDataset(Tensor(np.array(Train_img_data)), Tensor(np.array(train_class_name)))
torch_dataset_test = Data.TensorDataset(Tensor(np.array(Test_img_data)), Tensor(np.array(test_class_name)))


# Defining trainloader and test loader
trainloader = torch.utils.data.DataLoader(torch_dataset_train, batch_size= 8, shuffle = True) # DataLoader is an iterator that iterates the iamges in batches for the model
testloader = torch.utils.data.DataLoader(torch_dataset_test, batch_size= 8, shuffle = True)



torch_dataset_train = Data.TensorDataset(Tensor(np.array(Train_img_data)), Tensor(np.array(train_class_name)))
torch_dataset_test = Data.TensorDataset(Tensor(np.array(Test_img_data)), Tensor(np.array(test_class_name)))


# shape of training data
dataiter = iter(trainloader)
images = next(dataiter)
images[0].shape

# defining the model architecture
class CNNNet(nn.Module):
  def __init__(self):
      super(CNNNet, self).__init__()

      self.cnn_layers = nn.Sequential(
          # Defining a 2D convolution layer
          nn.Conv2d(3, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(16, 3, kernel_size=(50, 50), stride=(1, 1)),
          nn.MaxPool2d(kernel_size=1, stride=1, padding=0, ceil_mode=False)
      )

      self.linear_layers = nn.Sequential(
          nn.Linear(3, 3)
      )

  # Defining the forward pass
  def forward(self, x):
      x = self.cnn_layers(x)
      x = x.view(x.size(0), -1)
      x = self.linear_layers(x)
      return x


# Define the optimizer and loss function
# Defining the model
model = CNNNet()

# Defining the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# Defining the loss function
criterion = nn.CrossEntropyLoss()

# Checking if GPU is available
print(torch.cuda.is_available())
if torch.cuda.is_available():
    model = model.to("cuda")
    criterion = criterion.to("cuda")

print(model)


!export CUDA_LAUNCH_BLOCKING=1


#train this model for 10 epochs
for i in range(10):

    running_loss = 0
    model.train() # indicator for training phase
    for images, labels in trainloader:

        if torch.cuda.is_available():
          images = images.to("cuda")
          labels = labels.to("cuda")

        # Training pass
        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)

        #This is where the model learns by backpropagating
        loss.backward()

        #And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(trainloader)))


# Save the model
filepath = ROOT_DIR + "model.pt"
torch.save(model.state_dict(), filepath)


model_trained = CNNNet()
model_trained.load_state_dict(torch.load(filepath))

# device = "cuda" # --> use if GPU is present

# [.2, .5, .3]
# y_pred_list = []
# y_true_list = []
# with torch.no_grad():
#     for x_batch, y_batch in testloader:
#         x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#         y_test_pred = model(x_batch)
#         print(y_test_pred)
#         _, y_pred_tag = torch.max(y_test_pred, dim = 1)
#         y_pred_list.extend(y_pred_tag.cpu().numpy())
#         y_true_list.extend(y_batch.cpu().numpy())


#prediction
y_pred_list = []
y_true_list = []
with torch.no_grad():
    for x_batch, y_batch in testloader:
        x_batch, y_batch = x_batch.to(), y_batch.to()
        y_test_pred = model(x_batch)
        print(y_test_pred)
        _, y_pred_tag = torch.max(y_test_pred, dim = 1)
        y_pred_list.extend(y_pred_tag.cpu().numpy())
        y_true_list.extend(y_batch.cpu().numpy())


# y_test
y_true_list_max = [m.argmax() for m in y_true_list]


# Accuracy of model
correct_count, all_count = 0, 0
for i in range(len(y_pred_list)):
    if(y_pred_list[i] == y_true_list_max[i]):
      correct_count += 1
    all_count += 1
print("\nModel Accuracy =", (correct_count/all_count))
