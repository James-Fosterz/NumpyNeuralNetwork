from time import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class DeepNeuralNetwork():
    def __init__(self, epochs, sizes):
        self.epochs = epochs
        self.model = self.create_model(sizes)
        
    def create_model(self, sizes):
        model = nn.Sequential(nn.Linear(sizes[0], sizes[1]),
                      nn.ReLU(),
                      nn.Linear(sizes[1], sizes[2]),
                      nn.ReLU(),
                      nn.Linear(sizes[2], sizes[3]),
                      nn.Sigmoid(),
                      nn.Linear(sizes[3], sizes[4]),
                      nn.LogSoftmax()
                      )
        return model
    
    # Train Part
    def train(self, train, test):
        ac_values = []
        loss_values = []
        loss = nn.NLLLoss()
        images, labels = next(iter(train))
        images = images.view(images.shape[0], -1)
        
        #predicted images outcome
        predicted_val = self.model(images)
        #calculate the loss/how far off the prediction
        #used later in the backward propagation
        gradient = loss(predicted_val, labels)
        
        
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)#, momentum=0.9)
        time0 = time()
        #epochs = 5

        for e in range(self.epochs):
            running_loss = 0
            for images, labels in train:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)
            
                # Training pass
                optimizer.zero_grad()
                
                predicted_val = self.model(images)
                gradient = loss(predicted_val, labels)
                
                #This is where the model learns by backpropagating
                gradient.backward()
                
                #And optimizes its weights here
                optimizer.step()
                
                running_loss += gradient.item()
            else:
                print("Epoch {} - Training loss: {}".format(e, running_loss/len(train)))   
                ac_values.append(self.accuracy(test))
                loss_values.append(running_loss/len(train))
           
                
            
        print("\nTraining Time (in minutes) =",(time()-time0)/60)
        return  ac_values, loss_values
    
    def view_classify(self, img, ps):
        #Function for viewing an image and it's predicted classes.

        ps = ps.cpu().data.numpy().squeeze()
        
        fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
        ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
        ax1.axis('off')
        ax2.barh(np.arange(10), ps)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(10))
        ax2.set_yticklabels(np.arange(10))
        ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()
    
    # Testing function    
    def accuracy(self, test):
        images, labels = next(iter(test))
        img = images[0].view(1, 784)
        
        with torch.no_grad():
            predicted_vals = self.model(img)
        
        ps = torch.exp(predicted_vals)
        probab = list(ps.numpy()[0])
        print("Predicted Digit =", probab.index(max(probab)))
        self.view_classify(img.view(1, 28, 28), ps)
        
        correct_count, all_count = 0, 0
        for images,labels in test:
          for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                predicted_vals = self.model(img)
        
            
            ps = torch.exp(predicted_vals)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
              correct_count += 1
            all_count += 1
            
        print("Number Of Images Tested =", all_count)
        print("\nModel Accuracy =", (correct_count/all_count)*100)  
        
        return (correct_count/all_count)


def run():
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    #transforms.Normalize((0.1307,), (0.3081,))
                ])       
        
    train = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform),batch_size=100, shuffle=True)
    test = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transform), batch_size=100, shuffle=True)
    
    
    sizes = [784, 128, 64, 32, 10]
                          
    
    dnn = DeepNeuralNetwork(60, sizes)
    
    return dnn.train(train, test)
    
acc, loss = run()


plt.plot(acc, loss)
plt.show()
print(acc)