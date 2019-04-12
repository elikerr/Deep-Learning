## Chest Xray Data

# The goal of this Assignment was to take existing deep learning architectures and apply them to real
# life imaging challenges. In this assignment we used chest xray data to train and test our models
# Each point in this dataset corresponds to a chest xray image as well as a vector consisting of 14 binary 
# classifications.


import torch.nn as nn
# This block of code imports an 18 layer ResNet model from the pytorch library.
# Loads resnet18 network and modifies last layer to fit our classification space
network = models.resnet18(True)
network.fc = nn.Linear(in_features=512, out_features=14, bias=True)

# Binary Cross Entropy loss with Logits imported from pytorch to match target vector.
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(network.parameters(),lr= .0001)

#Set up loaders for batch size and validation (Data loaders given and predefined)
train_loader = torch.utils.data.DataLoader(train_dataset_ex1, shuffle = True, batch_size = 16, num_workers = 8)
val_loader = torch.utils.data.DataLoader(val_dataset_ex1, batch_size = 16, num_workers = 8)

#puts network onto GPU
network = network.cuda()

#this block will train my model
for epoch in range(3):
  
    network.train() #Toggles model into training mode
    
    losses = []
    
    for images, targets in train_loader: # 1)
      
        optimizer.zero_grad() # 3)
        
        #putting variables on GPU since model is on GPU
        images = images.cuda()
        targets = targets.cuda()
        
        #running each model by adapting the imagees tensor to the expected input size of each model
        out = network(images)
        
        #calculating the losses with the defined criterion
        loss = criterion(out, targets)
        
        loss.backward() #Backprop
        
        optimizer.step() #Stachastic Gradient descent
        
        losses.append(loss.item()) 
        
    print('Epoch ' + str(epoch+1)) 
    
# An analysis of my model on the validation set. 
scores= get_score_model_chestxray_binary_model(network, val_loader)
score=np.mean(scores)
print('Validation AUC: ' + str(score))

# An analysis of my model on the test set. 
test_loader=torch.utils.data.DataLoader(test_dataset_ex1, batch_size = 16, num_workers = 8)
scores= get_score_model_chestxray_binary_model(network, test_loader)
text_scores = ['{:.4f}'.format(score) for score in scores] 
labels=train_dataset_ex1.get_labels_name()
max_auc=max(scores)
min_auc=min(scores)
index1=scores.index(max_auc)
index2=scores.index(min_auc)
print('The Class with the best AUC score is '+labels[index1]+' with an AUC of '+'{:.4f}'.format(max_auc))
print('The Class with the best AUC score is '+labels[index2]+' with an AUC of '+'{:.4f}'.format(min_auc))
score=np.mean(scores)
print('The average AUC score for the test data set: '+'{:.4f}'.format(score))








## Pneumonia detection challenge

#First I will generate my model as it we are directed. I've modified the two final layers of the 
# ResNet-18 model to seqential layers. This causes the model to output a 14 x 14 grid.
model3 = models.resnet18(True)
model3.layer4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=False)
model3.avgpool = nn.Sequential()
model3.fc = nn.Sequential()

#define Loss function
criterion4 = nn.BCEWithLogitsLoss()
optimizer4 = torch.optim.Adam(model3.parameters(),lr= .0001)
#Set up loaders for batch size and validation
train_loader4 = torch.utils.data.DataLoader(train_dataset_ex2, shuffle = True, batch_size = 16, num_workers = 8)
val_loader4 = torch.utils.data.DataLoader(val_dataset_ex2, batch_size = 16, num_workers = 8)

model3 = model3.cuda()

#this block will train my model
for epoch in range(3):
  
    model3.train() #Toggles model into training mode
    
    losses4 = []
    
    for images, labels, targets in train_loader4: # 1)
      
        optimizer4.zero_grad() # 3)
        
        #putting variables on GPU since model is on GPU
        images = images.cuda()
        labels = labels.cuda()
        targets = targets.view(targets.size(0),-1)
        targets = targets.cuda()
        
        #running each model by adapting the imagees tensor to the expected input size of each model
        out4 = model3(images)
        
        #calculating the losses with the defined criterion
        loss4 = criterion4(out4, targets)
        
        loss4.backward()
        
        optimizer4.step() 
        
        losses4.append(loss4.item())
        
    print('Epoch ' + str(epoch+1)) 
    
    #An analysis of my model on the validation set. 
    scores= get_score_model_pneumonia_location_model(model3, val_loader4)
    score=np.mean(scores)
    print('Validation AUC: ' + str(score))


##this block will test my model
test_loader=torch.utils.data.DataLoader(test_dataset_ex2, batch_size = 16, num_workers = 8)
model3 = model3.cuda()

with torch.no_grad():
    correctpredictions = []
    correct = []
    incorrectpredictions = []
    incorrect = []
    index = 0
    for images, labels, targets in test_loader: # 1)

        #putting variables on GPU since model is on GPU
        images = images.cuda()
        labels = labels.cuda()
        targets = targets.view(targets.size(0),-1)
        targets = targets.cuda()
        size = targets.shape[0]
        predictions = model3(images)
        out = predictions.cpu()
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        predictions = (predictions > 0).astype('float32')
        predictions = predictions.reshape((size,14,14))
        targets = targets.reshape((size,14,14))
        index2=0
        
        for prediction in predictions:
            if (prediction == targets[index2]).all():
                correctpredictions.append(out[index2])
                correct.append(test_dataset_ex2[index])
                index+=1
                index2+=1
            else:
                incorrectpredictions.append(out[index2])
                incorrect.append(test_dataset_ex2[index])
                index+=1
                index2+=1
        
print('\nA Few Correct Predictions: ')
#plot a few images from the dataset
# reddish areas mean ones in the grid (pneumonia presence)
#blueish areas mean zeros in the grid (pneumonia absence)
plot_grid_over_xray(correct[1])
plot_grid_over_xray(correct[0])

print('/nA Few Incorrect Predicitions: ')
plot_grid_over_xray(incorrect[0])
plot_grid_over_xray(incorrect[1])
