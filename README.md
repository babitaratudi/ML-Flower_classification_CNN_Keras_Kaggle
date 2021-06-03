# Flower Clssification with Deep CNN by using Keras Library

## Importing Various Modules from Keras Library.
* Dataset downloaded from kaggel.
* Different flower saved in the form of an array.
* By using keras library downloaded Deep learning library sunch as models,optimizer,layers.

## Making training, test and validation data
* Each of the images has been resized to 150 * 150 pixel format.
* Made the function to make train data set directory for each flower.
* Visualized some random images.
* Each of the image is devided by 255 so as to get values between 0 and 1.
* Splitting into Training and Validation Sets. 
* Setting the Random Seeds.

## Modelling
* Building the ConvNet Model by initialization of the Sequential model.
* Then we have used 4 convolutional layers with having input shape 150 * 150 pixel for our model, activation function as ReLU and 32 filters of kernel size 5 * 5, 64 filters of kernel size 3 * 3, 96 filters of kernel size 3 * 3 and 96 filters of kernel size 3 * 3 respectively in each convolution layer.
* Two Max pooling layers of size 2 * 2 units have been used in each layer.
* Next the flattening operation is performed to convert the pooled features into a single vector.
* This flattened vector is fed into a hidden layer with 256 neurons which applies the ReLU activation function.
* Since we have more than 2 categories, we are using softmax activation function.
* Data Augmentation to prevent Overfitting.

## Compiling,summery and Fitting the model to the training set
* We compile the model using the Adam optimizer with learning rate = 0.001 optimizer, loss function categorical_crossentropy and metrics as accuracy.
* At last we fit our model to the classifier with 25 epochs.

## Evaluating the Model Performance and Prediction
* Plotted graph of loss and validation loss.
* After that downloaded one image for testing purpose and got the result with with right array index of that particular flower.
