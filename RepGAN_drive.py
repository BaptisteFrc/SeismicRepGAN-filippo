## -*- coding: utf-8 -*-
#!/usr/bin/env python3
u"""General informations"""
import datetime
__author__ = "Giorgia Colombera, Filippo Gatti"
__copyright__ = "Copyright 2021, CentraleSupélec (MSSMat UMR CNRS 8579)"
__credits__ = ["Giorgia Colombera,Filippo Gatti"]
__license__ = "GPL"
__version__ = "1.0.1"
__Maintainer__ = "Filippo Gatti"
__email__ = "filippo.gatti@centralesupelec.fr"
__status__ = "Beta"


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from RepGAN_utils import *
from interferometry_utils import *
import math as mt

import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float32')
gpu = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu))
tf.config.experimental.set_memory_growth(gpu[0], True)

import MDOFload as mdof
from plot_tools import *

from RepGAN_model import RepGAN
import RepGAN_losses

import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="RepGAN",

    # track hyperparameters and run metadata with wandb.config
    config={
        "epoch": 20,
        "batch_size": 128
    }
)

def Train(options):

    with tf.device(options["DeviceName"]):
        
        losses = RepGAN_losses.getLosses(**options)
        optimizers = RepGAN_losses.getOptimizers(**options)
        callbacks = RepGAN_losses.getCallbacks_wdb(**options)

        # Instantiate the RepGAN model.
        GiorgiaGAN = RepGAN(options)

        # Compile the RepGAN model.
        GiorgiaGAN.compile(optimizers, losses, metrics=[tf.keras.metrics.Accuracy()])

        
        # Build shapes
        # GiorgiaGAN.build(input_shape=(options['batchSize'],options['Xsize'],options['nXchannels']))

        # Build output shapes
        GiorgiaGAN.compute_output_shape(input_shape=(options['batchSize'],options['Xsize'],
                                options['nXchannels']))


        if options['CreateData']:
            # Create the dataset
            train_dataset, val_dataset = mdof.CreateData(**options)
        else:
            # Load the dataset
            #train_dataset, val_dataset = mdof.LoadData(**options)
            train_dataset, val_dataset = mdof.Load_Un_Damaged(0, **options)
        
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Train RepGAN
        history = GiorgiaGAN.fit(x=train_dataset,batch_size=options['batchSize'],
                                 epochs=options["epochs"],
                                 callbacks=[WandbMetricsLogger(log_freq='batch'),WandbModelCheckpoint(filepath='checkpoint/1/',save_freq=100)],
                                 validation_data=val_dataset,shuffle=True,validation_freq=100)

        

        DumpModels(GiorgiaGAN,options['results_dir'])

        # PlotLoss(history,options['results_dir']) # Plot loss


def Evaluate(options):

    with tf.device(options["DeviceName"]):

        losses = RepGAN_losses.getLosses(**options)
        optimizers = RepGAN_losses.getOptimizers(**options)
        callbacks = RepGAN_losses.getCallbacks(**options)

        # Instantiate the RepGAN model.
        GiorgiaGAN = RepGAN(options)

        # Compile the RepGAN model.
        GiorgiaGAN.compile(optimizers, losses, metrics=[
                           tf.keras.metrics.Accuracy()])
        # Build output shapes
        GiorgiaGAN.compute_output_shape(input_shape=(options['batchSize'], options['Xsize'],
                                                     options['nXchannels']))
        
        latest = tf.train.latest_checkpoint(options["checkpoint_dir"])
        
        GiorgiaGAN.load_weights(latest)

        if options['CreateData']:
            # Create the dataset
            train_dataset, val_dataset = mdof.CreateData(**options)
        else:
            # Load the dataset
            #train_dataset, val_dataset = mdof.LoadData(**options)
            train_dataset, val_dataset = mdof.Load_Un_Damaged(0,**options)
        
        # Re-evaluate the model
        import pdb
        pdb.set_trace()
        loss, acc = GiorgiaGAN.evaluate(val_dataset)
    

if __name__ == '__main__':
    options = ParseOptions()
    
    if options["trainVeval"].upper()=="TRAIN":
        Train(options)
    elif options["trainVeval"].upper()=="EVAL":
        Evaluate(options)