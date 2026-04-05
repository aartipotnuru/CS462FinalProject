import pandas as pd
import torch.nn as nn
import torch

from sklearn.metrics import accuracy_score

from torch.utils.data import TensorDataset, DataLoader


#one for training, one for the testing dataset, and one for validation

#to test locally update these variables to the processed data path names
trainingDataPath = ""
testingDataPath = ""
dataFromValidationPath = ""

trainingData = pd.read_csv(trainingDataPath)

testingData = pd.read_csv(testingDataPath)

dataFromValidation = pd.read_csv(dataFromValidationPath)


#tensors for each feature that is being inputted and each label
#where ingredient_category_id represents the categories from 0 through 9
firstTraining = torch.tensor(trainingData[["ingredient_category_id"]].values, dtype = torch.float32)
firstTesting = torch.tensor(testingData[["ingredient_category_id"]].values, dtype = torch.float32)
validationFirst = torch.tensor(dataFromValidation[["ingredient_category_id"]].values, dtype = torch.float32)

#these labels are present to see what prediction will be outputted
secondTraining = torch.tensor(trainingData["substitution_category_id"].values, dtype = torch.long)
secondTesting = torch.tensor(testingData["substitution_category_id"].values, dtype = torch.long)
validationSecond = torch.tensor(dataFromValidation["substitution_category_id"].values, dtype = torch.long)

#load the data with 32 as batch size at first
loadForTraining = DataLoader(TensorDataset(firstTraining, secondTraining), batch_size = 32, shuffle = True)
loadForValidation = DataLoader(TensorDataset(validationFirst, validationSecond), batch_size = 32)
value = secondTraining.unique().shape[0]


#creating model with a neural network

#first layer : 64 neurons are present then the function for activation with ReLU()
#dropping the results to ensure the model does not overfit into its category

model = nn.Sequential(
    nn.Linear(1,64),
    nn.ReLU(),
    nn.Dropout(0.1),

    #second layer and repeats ReLU() and dropout
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Dropout(0.1),

    #third layer
    nn.Linear(128, 64),
    nn.ReLU(),

    #where each category has an output
    nn.Linear(64, value)
)

#calculating the loss and incorporating Adam to recalculate the weights to prevent further losses
entropyLoss = nn.CrossEntropyLoss()

optimized = torch.optim.Adam(model.parameters(), lr = 0.001)


#training aspect 35 full passes through the entire dataset that is present

for e in range(35):
    amt = 0
    model.train()

    for i, j in loadForTraining:
        optimized.zero_grad()

        #performing the forward pass
        prediction = model(i)
        reduce = entropyLoss(prediction, j)

        #optimized.zero_grad()
        reduce.backward()
        optimized.step()

        amt += reduce.item() #getting the total amount loss

    model.eval()
    #getting the accuruacy on the dataset
    with torch.no_grad():
         validationPred = model(validationFirst).argmax(dim = 1)
    results = accuracy_score(validationSecond, validationPred)

    print(f"Epoch number {e + 1} | Loss: {amt: .4f} | Validation: {results: .4f}")
#evaluting each part


#print("accuracy: ", result)


model.eval()

#training on the test dataset
with torch.no_grad():
     values = model(firstTesting).argmax(dim = 1)

result = accuracy_score(secondTesting, values)

print(f"Testing accuracy: {result: .4f}")



