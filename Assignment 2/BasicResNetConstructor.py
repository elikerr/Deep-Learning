'''
First, i need to consruct my model.
to construct my model, i will use the pytorch master documentation  
as a general structure guide and modify the network constructor and 
block requirements to fit the criteria provided.
'''

import math
import torch.utils.model_zoo as model_zoo

#these will be the building blocks of the different layers of my network
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
#Basic Block built (conv -> norm -> relu)*2, no pooling
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
       
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

    
#This is the constructor for the first neural network that I will be testing
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=14):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2,
                               bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.layer1 = self._make_layer(block, 64, layers, stride=2)

        self.avgpool = nn.AvgPool2d(3, stride=2)
        self.fc = nn.Linear(10816, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
    
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

#Now that i have my different layer features and constructor designed, i can start generating my model
#I can only fit 4 blocks before going over 500k parameters.
model = ResNet(BasicBlock, 4)
print(count_number_parameters(model))

#define Loss function
criterion2 = nn.BCEWithLogitsLoss()
optimizer2 = torch.optim.Adam(model.parameters(),lr= .0001)
#Set up loaders for batch size and validation
train_loader2 = torch.utils.data.DataLoader(train_dataset_ex1, shuffle = True, batch_size = 16, num_workers = 8)
val_loader2 = torch.utils.data.DataLoader(val_dataset_ex1, batch_size = 16, num_workers = 8)

#Now i need to train and test my model
model = model.cuda()

#this block will train my model
for epoch in range(5):
  
    model.train() #Toggles model into training mode
    
    losses2 = []
    
    for images, targets in train_loader2: # 1)
      
        optimizer2.zero_grad() # 3)
        
        #putting variables on GPU since model is on GPU
        images = images.cuda()
        targets = targets.cuda()
        
        #running each model by adapting the imagees tensor to the expected input size of each model
        out2 = model(images)
        
        #calculating the losses with the defined criterion
        loss2 = criterion(out2, targets)
        
        loss2.backward()
        
        optimizer2.step() 
        
        losses2.append(loss2.item()) 

        
    print('Epoch ' + str(epoch+1)) # 6)
    
    # An analysis of my model on the validation set. 
    scores2= get_score_model_chestxray_binary_model(model, val_loader2)
    score2=np.mean(scores2)
    print('Validation AUC: ' + str(score2))

# Now I will test a similar structure with a maxpooling layer built into the block
class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock2, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
       
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.maxpool(out)

        return out

#Now that i have my different layer features and constructor designed
#i can start generating my model
model2 = ResNet(BasicBlock2, 4)
model2.fc = nn.Linear(in_features=5184, out_features=14)

print(count_number_parameters(model))
#print(model)

#define Loss function
criterion3 = nn.BCEWithLogitsLoss()
optimizer3 = torch.optim.Adam(model2.parameters(),lr= .0001)
#Set up loaders for batch size and 
train_loader3 = torch.utils.data.DataLoader(train_dataset_ex1, shuffle = True, batch_size = 16, num_workers = 8)
val_loader3 = torch.utils.data.DataLoader(val_dataset_ex1, batch_size = 16, num_workers = 8)

#Now i need to train and test my model
model2 = model2.cuda()

#this block will train my model
for epoch in range(5):
  
    model2.train() #Toggles model into training mode
    
    losses3 = []
    
    for images, targets in train_loader3: # 1)
      
        optimizer3.zero_grad() # 3)
        
        #putting variables on GPU since model is on GPU
        images = images.cuda()
        targets = targets.cuda()
        
        #running each model by adapting the imagees tensor to the expected input size of each model
        out3 = model2(images)
        
        #calculating the losses with the defined criterion
        loss3 = criterion(out3, targets)
        
        loss3.backward() #Backprop
        
        optimizer3.step() #Stachastic Gradient descent
        
        losses3.append(loss3.item())
        
    print('Epoch ' + str(epoch+1)) 
    
    # An analysis of my model on the validation set. 
    scores3= get_score_model_chestxray_binary_model(model2, val_loader3)
    score3=np.mean(scores3)
    print('Validation AUC: ' + str(score3))