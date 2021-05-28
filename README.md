## Flower Clssification using Deep CNN by using Keras Library

# Importing Various Modules from Keras Library.
*Dataset downloaed from kaggel.
*Different flower saved in the form of an array.
*By using keras library downloaded Deep learning library sunch as models,optimizer,layers.

# Making training, test and validation data
*Each of the images has been resized to 150 * 150 pixel format.
*Made the function to make train data set directory for each flower.
*Visualized some random images.
*Each of the image is devided by 255 so as to get values between 0 and 1.
*Splitting into Training and Validation Sets.
*Setting the Random Seeds.

# Modelling
*
Here we have first initialised the Sequential model.
Then we have used 2 convolutional layers having input shape 128 X 128 oxels for our model, activation function as ReLU and 32 feature detector of size 3 * 3.
Two Max pooling layers of size 2 * 2 units have been used.
Next the flattening operation is performed to convert the pooled features into a single vector.
This flattened vector is fed into a hidden layer with 256 neurons which applies the ReLU activation function.
In the end we get the output as one of the flowers from 102 categories. Since we have more than 2 categories, we are using softmax activation function.
#Compiling and Fitting the model to the training set
We compile the model using the Adam optimizer with learning rate = 0.001 optimizer, loss function categorical_crossentropy and metrics as accuracy.
At last we fit our model to the classifier with 40 epochs.
