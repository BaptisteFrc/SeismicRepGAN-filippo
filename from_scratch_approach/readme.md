# Certification and Robustness of Neural Networks by Gaussian Smoothing

## Description of the project

Our heritage is a treasure that must be protected. In the context of historic buildings, where renovation can be difficult without damaging their historical wealth, and where it is vital to enhance their value, particular attention is being paid to monitoring. This involves, for example, using the data supplied by sensors to feed digital twins in order to prevent rather than cure potential disasters. The stakes are all the higher in the context of our project, i.e. in the case of natural disasters, and earthquakes in particular.

Monitoring means predicting. Predicting the rate of damage to a building, predicting the response of another building to seismic excitation. Our project deals with the implementation of neural networks to perform the role of such predictors. On the other hand, the field of historic buildings suffers from a cruel lack of data. We sometimes have no information about the past of an old building. This makes it particularly difficult to train networks. This gap can be filled by recent advances in generative AI. By generative, we mean being able to produce data that is entirely plausible but totally artificial. This data can then be used to train other networks. The problem of protecting historic buildings therefore revolves around these two words: predict and generate.

Although both are currently carried out independently, the aim of the project is to continue writing the code supplied by the customer in order to address both issues simultaneously.

This project will seek to justify the use of LSTM layers in the RepGAN generator auto-encoder structure in order to implement them and thus add the predictive dimension to the client code.

## Implementing rules

This code uses very well known libraries like torch, numpy and matplotlib. We also use h5py to read our dataset that is under the HDF5 format.

## How to use ?

the model_tested file contains only the 6 auto-encoders used for the various tests in the study. The training file handles all the data management, from creation to processing, as well as training the networks and saving the weights. The test file is the link with wandb.

## Support

If you need any further information, you can contact us at this email adress : aurele.gallard@student-cs.fr

## Credits

This project has been carried out in the context of CentraleSupelec AI project pole. Our team was composed of 5 first year students :
Mingyang Jacques Sun, Mathilde Morhain, Yi Zhong, Baptiste François and Aurèle Gallard ; in partnership with a LMPS researcher : Filippo Gatti.

## Licence

GPL

## Projet Status

This project, in its academic context, has come to its end. But it is to be pursued from a different angle.