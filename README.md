# deep_learning_projs
Deep Learning Algorithms Practice

# 2. Model comparison for MNIST data
   This section compares preformance between DNN, CNN, VGG, and ResNet models in labeling the MNIST dataset.

## Data Used: MNIST dataset 
Each data sample has 785 pieces of data. The first column, called ‘label’, is the label of what number is in the image. It ranges from 0 to 9.3 The other 784 columns are individual pixels. They range from 0 to 255, for the grey scale of the image. The column names are ‘pix-(row number)-(column number)’. There is no data missing from this dataset.

The data is split 60/20/20 for training/validation/testing. This data set is fairly clean, so the only pre-processing is normalization of the data using a min-max normalization since this data does not have any outliers. The min and max for the label column will be 0 and 9, respectively. The min and max for the pixel columns will be 0 and 255, respectively.

## Model Overviews:
### DNN Model
The DNN model has six layers. The first layer flattens the two-dimensional image data into a one-dimensional array. The other four layers step the dimensionality of the flattened array (length: 784) down by powers of two using a ReLu activation layer. The final layer steps the dimensionality down to the needed output length of ten and uses a softmax activation to get a probability for multiple classifications. 

### CNN Model
The CNN model has 2 blocks consisting of a convolution layer and max pooling. The final block is flattening and a dense layer to get a one-dimensional array of length ten. ReLu was used as the activation function for the CNN layers. Softmax was used as the final activation in order to ensure that outputs are probabilities for multi-classification. 

### VGG Model
The VGG model is used to classify images of size 224 x 224. It’s made up of VGG blocks, consisting of multiple CNN layers followed by a max-pooling layer. The CNN layers include padding to maintain a consistent shape across the block. VGG-A consists of 5 VGG blocks before switching to 3 dense layers to get the output, 11 weight layers in total. This model has 132,872,194 trainable parameters. The MNIST images are 28 x 28. To accommodate this and to make the model trainable in a reasonable amount of time, the VGG for this report has one fewer VGG blocks and reduces the depth proportionally to the reduced input size, resulting in 418,806 trainable parameters.

## Optimization

The DNN and CNN models were run with three different optimizers to compare the difference. The optimizers perform differently given different learning rates, so each was run with three different learning rates. Tables 2 and 3 show the results. The first optimizer, Stochastic Gradient Descent (SGD), performed well at lower learning rates. The SGD can get stuck on local minimums and is susceptible to gradient vanishing. 
The second optimizer tested was Adagrad. Adagrad uses momentum to reduce the risk of being stuck on a local minimum. It can tend to assign a higher learning rate to infrequent features. That may be a benefit in this instance because the majority of values in the image are 0, where there is whitespace. Adagrad was the highest performing optimizer with a learning rate of 0.1. 

The last optimizer tested was Adam. This is known to be a great performing optimizer. It performed the best at a learning rate of 0.001. It performed poorly at higher learning rates. This makes sense because a large learning rate can overpower an optimizer trying to find a minimum value.

Adagrad with a learning rate of 0.1 was used for the model comparison. When comparing learning rates in the results section, the optimizer that performed best at the corresponding learning rate was used.

## Usage

The model to run (input number of corresponding model):   
	1:DNN,  
	2:CNN,  
	3:VGG,  
	4:Resnet

```python
# runs DNN model
py trf_dl_hw3 1
```

 # 3. Medical Image Segmentation

 ## Data Used: Retina Blood Vessel Segmentation dataset
The data set for medical image segmentation consists of 100 retina scan images and masks, 20 test and 80 train images and masks. The masks are the labels for this data set. The goal is to replicate these masks using only the image as the data input. Each image is 512 x 512 x 3. The last dimension shows that there are RGB values for the images. Each mask is 512 x 512 x 1. The last dimension shows that the mask is greyscale. All values in the images and masks are in the range 0 to 255. There is no missing data. The training data will be split 80/20 for training and validation and the data labeled as testing will be used for testing. The images and masks are loaded using PILLOW and are normalized by dividing all values by 255. No other pre-processing is used.

This data set is used for early detection of eye disease by mapping retinal blood vessels using retinal eye scans.

## Model Overiew

The U-Net Model is the basis for this assessment. A comparison was done across the following models: U-Net 2 layers, U-Net 3 layers, U-Net 4 layers, U-Net 5 layers. Binary crossentropy was used as the loss function after a comaprison between the following loss functions: Binary Crossentropy, Dice, Cosine-similarity, MSR. Adam was used as the optimizer after a comaprison across the following optimizers: SGD, Adam, RMSprop, Adagrad.

## Usage

Run HW3 Retina Scan with U-model: tf_dl_hw3
tf_dl_hw3 runs all of the models needed for Deep Learning HW3.
The model to run (all run for each execution):   
	U-Net 2 layers,
	U-Net 3 layers,
	U-Net 4 layers,
	U-Net 5 layers,  
	
The learning rate has a default value of 0.01.
The epoch number has a default value of 50.

```python
# runs training for all models
py trf_dl_hw3 train ./[filepath that contains the Data folder]

# runs testing for all models using saved weights
py trf_dl_hw3 test ./[filepath that contains the Data folder]
```
