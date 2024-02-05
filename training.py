#Importation of the useful libraries
import glob
import tensorflow as tf
import numpy as np
import pandas as pd
from os.path import join as opj
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import h5py
from matplotlib import pyplot as plt
import csv
import torch
from torchvision import datasets
from torchvision import transforms
import scipy
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.colors import ListedColormap

#Importation of the models
from models_tested import resnet_AE, lstm_resnet_AE, lstm_conv_AE, conv_AE, linear_encoder, linear_decoder

#Use of wandb for visualization
import wandb
import pprint
wandb.login()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Loading the data
store_dir="input_data"
def LoadData():

    fid_th = opj(store_dir,"Data2.h5")

    h5f = h5py.File("Data2.h5",'r')
    X = h5f['X'][...]
    h5f.close()

    # Split between train and validation set (time series and parameters are splitted in the same way)
    Xtrn, Xvld = train_test_split(X, random_state=0, test_size=0.1)

    return Xtrn, Xvld

Xtrn, Xvld=LoadData()

#Data processing
def vect_sigm(tab) :
  M,N,P=tab.shape
  res=np.zeros((M,N,P))
  for i in range(M) :
    for j in range(N) :
      for k in range(P) :
        res[i,j,k] = 1/(1+np.exp(-tab[i,j,k]))
  return res

Xvld = vect_sigm(Xvld)
Xtrn = vect_sigm(Xtrn)
xvld = torch.tensor(Xvld[:,20:261,:]).to(torch.float32)
xtrn = torch.tensor(Xtrn[:,20:261,:]).to(torch.float32)

#Data visualization
fig, ax = plt.subplots(figsize=(8,4))
colors = cm.rainbow(np.linspace(0, 1, xtrn.shape[0]))
for i, (yp, c) in enumerate(zip(xtrn, colors)):
    ax.plot(yp, linewidth=0.5, color=c)
ax.set_xlabel("$t [s]$")
ax.set_ylabel(r"$x_g(t)$")
ax.set_title('Input ground motion')

#Function to add zeroes to the vector in input of the LSTM layer
def fill_with_zeros(x) :
    if len(x.size())==3 :
      n,m,p=x.size()
      res=torch.Tensor(np.zeros((n,m,p+5)))
      for i in range(n) :
        for j in range(m) :
          res[i,j,:-5]=x[i,j,:]
    else :
      n,m=x.size()
      res=torch.Tensor(np.zeros((n,m+5)))
      for i in range(n) :
        res[i,:-5]=x[i,:]
    return res.to(device)

# Hyperparameters values for Conv and Resnet models
input_size = 4
hidden_size = np.array([[8, 16, 32], [32, 16, 8]])
output_size = 64
kernel_size = 3
stride = 2
padding = 1

#Definition of sweep parameters to test several values
sweep_config = {'name':'conv_lstm',
    'method': 'grid'
    }

parameters_dict = {
    'epochs': {
        'values': [1200]
        },
    'batch_size': {
          'values': [16]
        },
    'learning_rate':{'value': 1e-3},
    }

sweep_config['parameters'] = parameters_dict
pprint.pprint(sweep_config)
sweep_id = wandb.sweep(sweep_config, project="project_name")

#Definition of the train function (entity = your wandb account)
def train(config=None) :
  with wandb.init(config=config, entity='wandb_account') :
    config=wandb.config

    # Model Initialization (choose the model you want to train)
    #Model conv  
    model = conv_AE(input_size, hidden_size, output_size, kernel_size, stride, padding)
    #Model Linear
    #linear_encoder().reset_parameters()
    #linear_decoder().reset_parameters()
    #lstm_AE = torch.nn.Sequential(lstm_encoder(),lstm_decoder())
    #model = lstm_AE
    #Model Resnet LSTM
    #model = lstm_resnet_AE(input_size, hidden_size, output_size, kernel_size, stride, padding)
      
    model = model.to(device)

    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()

    # Using an Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = config.learning_rate,
                            weight_decay = 1e-8)

    epochs = config.epochs
    losses = []
    test_losses = []

    loader = torch.utils.data.DataLoader(dataset = xtrn,
                                        batch_size = config.batch_size,
                                        shuffle = True)

    test_loader = torch.utils.data.DataLoader(dataset = xvld,
                                              batch_size = 32,
                                              shuffle = True)

    for epoch in range(epochs):
        model.train()
        train_loss=0
        for time_s in loader:
          time_s=time_s.to(device)
          # Reshaping the image to (-1, 784)
          time_s_2 = time_s[:,0:161,:].to(device)

          # Output of Autoencoder
          reconstructed = model(time_s_2)

          # Calculating the loss function
          loss = loss_function(reconstructed, time_s)

          # The gradients are set to zero,
          # the gradient is computed and stored.
          # .step() performs parameter update
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        # Storing the losses in a list for plotting
          train_loss+=torch.Tensor.cpu(time_s.size()[0]*loss.detach()).numpy()

        losses.append(train_loss/xtrn.size()[0])
        model.eval()
        test_loss=0
        for time_s in test_loader:
          time_s=time_s.to(device)
          time_s_2 = time_s[:,0:161,:]

          # Output of Autoencoder (the reconstruction is made from the cropped signal)
          reconstructed = model(time_s_2)

          # Calculating the loss function
          loss = loss_function(reconstructed, time_s)

          # Storing the losses in a list for plotting
          test_loss+=torch.Tensor.cpu(time_s.size()[0]*loss.detach()).numpy()
        test_losses.append(test_loss/xvld.size()[0])
      #Storing the losses on wandb
        wandb.log({"test loss": test_losses[-1], "train loss": losses[-1]})
    torch.save(model.state_dict(), f"my_model_conv_b{config.batch_size}_lr{config.learning_rate}_ep{config.epochs}.pth")
    wandb.save(f"my_model_conv_b{config.batch_size}_lr{config.learning_rate}_ep{config.epochs}.pth")

#Start of the training
wandb.agent(sweep_id, train, count=1)

#Loading the model (the model name must be changed according to the name chosen when it was created)
model = conv_AE(input_size, hidden_size, output_size, kernel_size, stride, padding)
best_model = wandb.restore('my_model_conv_b16_lr0.001_ep1200.pth', run_path="wandb_account/project_name/run_name")
model.load_state_dict(torch.load('my_model_conv_b16_lr0.001_ep1200.pth', map_location=torch.device('cpu')))
model.to(device)

#The prediction can be compared to the original signal for one sample
# time_s = xvld[3].to(device)
# time_s_2 = time_s[0:161,:]
# res=model(time_s_2).cpu().detach().numpy()
# plt.plot(time_s[:,0].cpu(), 'r')
# plt.plot(res[:,0], 'b')
