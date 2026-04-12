import pandas as pd
import torch.nn as nn
import torch
import numpy as np
import ast

from sklearn.metrics import accuracy_score

from torch.utils.data import TensorDataset, DataLoader

trainingDataPath = "data/processed/train.csv"
testingDataPath = "data/processed/test.csv"
dataFromValidationPath = "data/processed/val.csv"

trainingData = pd.read_csv(trainingDataPath)

testingData = pd.read_csv(testingDataPath)

dataFromValidation = pd.read_csv(dataFromValidationPath)

diets = ["vegan", "gluten_free", "nut_free", "dairy_free", "keto"]

embeddingAmt = 8

category = 10

#additional method that would be useful for tagging the specific dietaries

def tagging(num):
    try:
          specificDiet = ast.literal_eval(num) if isinstance (num, str) else (num or [])
    
    except Exception:
        specificDiet = []

    return [1.0 if i in specificDiet else 0.0 for i in diets]


#creating the feature
def feature(f):
    value = np.array([tagging(i) for i in f["ingredient_diets"]], dtype = np.float32)
    return value  #which would be the ingredients

#tensors for each feature that is being inputted and each label
#where ingredient_category_id represents the categories from 0 through 9
#long type for the embeddings
firstTraining = torch.tensor(trainingData["ingredient_category_id"].values, dtype = torch.long)
firstTesting = torch.tensor(testingData["ingredient_category_id"].values, dtype = torch.long)
validationFirst = torch.tensor(dataFromValidation["ingredient_category_id"].values, dtype = torch.long)

#these labels are present to see what prediction will be outputted
secondTraining = torch.tensor(trainingData["substitution_category_id"].values, dtype = torch.long)
secondTesting = torch.tensor(testingData["substitution_category_id"].values, dtype = torch.long)
validationSecond = torch.tensor(dataFromValidation["substitution_category_id"].values, dtype = torch.long)

#features needed for the tagging

tag1 = torch.tensor(feature(trainingData), dtype = torch.float32)
tag2 = torch.tensor(feature(dataFromValidation), dtype = torch.float32)
tag3 = torch.tensor(feature(testingData), dtype = torch.float32)


#load the data with 32 as batch size at first
loadForTraining = DataLoader(TensorDataset(firstTraining, tag1, secondTraining), batch_size = 32, shuffle = True)
loadForValidation = DataLoader(TensorDataset(validationFirst, tag2, validationSecond), batch_size = 32)
loadForTesting = DataLoader(TensorDataset(firstTesting, tag3, secondTesting), batch_size = 32)

value = secondTraining.unique().shape[0]


#creating model with a neural network

#first layer : 64 neurons are present then the function for activation with ReLU()
#dropping the results to ensure the model does not overfit into its category
#seperate class for model with the embeddings

class Model(nn.Module):
    def __init__(self, count, size, dietAmt, tot):
          super(Model, self).__init__()

          self.embed = nn.Embedding(tot, size)

          response = size + dietAmt

          self.layering = nn.Sequential(
            nn.Linear(response,64),
            nn.ReLU(),
            nn.Dropout(0.2),

            #second layer and repeats ReLU() and dropout
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

             #third layer
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, tot)
          )

        # now the forward is required
          
    def forward(self, first, second):
         amtOne = self.embed(first.squeeze()) #first batch size and second
         amtTwo = torch.cat([amtOne, second], dim = 1)
         return self.layering(amtTwo)

model = Model(
     count = category,
     size = embeddingAmt,
     dietAmt = 5,
     tot = value
)


#calculating the loss and incorporating Adam to recalculate the weights to prevent further losses
entropyLoss = nn.CrossEntropyLoss()

optimized = torch.optim.Adam(model.parameters(), lr = 0.001)


#training aspect 35 full passes through the entire dataset that is present

accuracy1 = 0
accuracy2 = 0
val = 10

for e in range(35):
    amt = 0
    model.train()

    for i, j, lbl in loadForTraining:
        optimized.zero_grad()

        #performing the forward pass
        prediction = model(i, j)
        reduce = entropyLoss(prediction, lbl)

        #optimized.zero_grad()
        reduce.backward()
        optimized.step()

        amt += reduce.item() #getting the total amount loss

    model.eval()
    #getting the accuruacy on the dataset
    with torch.no_grad():
         preds = []
         for i, j, lbl in loadForValidation:
            validationPred = model(i, j).argmax(dim = 1)
            preds.extend(validationPred.tolist())
    response = accuracy_score(validationSecond.tolist(), preds)

    print(f"Epoch number {e + 1} | Loss: {amt: .4f} | Validation: {response: .4f}")
#evaluting each part

   

#print("accuracy: ", result)
    
    if response > accuracy1:
        accuracy1 = response
        accuracy2 = 0
    else: 
        accuracy2 += 1
        if accuracy2 >= val:
            print(f"Stopping from this epoch {e + 1}")
            break

model.eval()

preds = []
#training on the test dataset
with torch.no_grad():
     for i, j, lbl in loadForTesting:
        values = model(i, j).argmax(dim = 1)
        preds.extend(values.tolist())

result = accuracy_score(secondTesting.tolist(), preds)

print(f"Testing accuracy: {result: .4f}")
print(f"Best validation result: {accuracy1: .4f}")
