import matplotlib.pyplot as plt
import keras
from datetime import datetime
import numpy as np
import os
from PIL import Image

#------------------------------------------------------------------------------
# Model Helper Functions
#------------------------------------------------------------------------------
# This function adds a datetime callback before each epoch

class TimeCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        epoch_begin = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("{time} ---".format(time=epoch_begin),end=(""))

def load_data(filepath,norm=True):
    filepath_img = filepath + "/image/"
    imgs = [Image.open(filepath_img + str(file_num)+".png") for file_num in range(len(os.listdir(filepath_img)))]


    filepath_mask = filepath + "/mask/"
    mask = [Image.open(filepath_mask + str(file_num)+".png") for file_num in range(len(os.listdir(filepath_mask)))]
    
    if norm:
        return np.array(imgs)/255, np.array(mask)/255
    else:
       return np.array(imgs), np.array(mask)

#------------------------------------------------------------------------------
# U-net specific functions
#------------------------------------------------------------------------------
def encode_block(filter, input_data,sub_layer_num, norm=True):
  """
  The contracting path follows the typical architecture of a convolutional network. It consists of the repeated
  application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU)
  and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number
  of feature channels.
  """
  sub_layer_num = str(sub_layer_num)

  conv1 = keras.layers.Conv2D(filter, kernel_size=(3, 3), padding='same', activation=None,name = "conv_"+sub_layer_num+"-1" )(input_data)
  if norm:
    conv1 = keras.layers.BatchNormalization()(conv1)
  
  conv2 = keras.layers.Conv2D(filter, kernel_size=(3, 3), padding='same', activation=None,name = "conv_"+sub_layer_num+"-2")(conv1)
  if norm:
    conv2 = keras.layers.BatchNormalization()(conv2)
  act = keras.layers.ReLU()(conv2)
  
  output_data = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = "maxpool_"+sub_layer_num)(act)
  skip_data = act

  return output_data, skip_data

def decode_block(filter, input_data, skip_con,sub_layer_num,norm=True):
    """
    Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2
    convolution ( up-convolution ) that halves the number of feature channels, a concatenation with
    the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions,
    each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution.
    """
    sub_layer_num = str(sub_layer_num)

    up_sample = keras.layers.Conv2DTranspose(filter, (2, 2), strides = 2, padding = 'valid', name="upsample_"+sub_layer_num)(input_data)

    comb = keras.layers.Concatenate(name="concat_"+sub_layer_num)([up_sample, skip_con])

    conv = keras.layers.Conv2D(filter, kernel_size=(3, 3), padding='same', activation=None, name="conv_up_"+sub_layer_num+"-1")(comb)
    if norm:
      conv = keras.layers.BatchNormalization()(conv)
    output_data = keras.layers.Conv2D(filter, kernel_size=(3, 3), padding='same', activation=None, name="conv_up_"+sub_layer_num+"-2")(conv)
    if norm:
      output_data = keras.layers.BatchNormalization()(output_data)
    output_data = keras.layers.ReLU()(output_data)

    return output_data

def build_u_model(num_layers, filter_num, norm=True):
  
  # Input
  Unet_input = keras.Input(shape=(512, 512, 3))
  print("Unet shape: ", Unet_input.shape)
  block_in = Unet_input
  skip_layers = []

  # Encode
  for i in range(num_layers-1):
    block_out, skip_layer = encode_block(filter_num, block_in,i+1, norm)
    skip_layers.append(skip_layer)
    filter_num = filter_num * 2
    block_in = block_out
  
  # Bottleneck
  conv1 = keras.layers.Conv2D(filter_num, kernel_size=(3, 3), padding='same', activation='relu', name = "conv_"+str(num_layers)+"-1")(block_in)
  block_in = keras.layers.Conv2D(filter_num, kernel_size=(3, 3), padding='same', activation='relu', name = "conv_"+str(num_layers)+"-2")(conv1)
  filter_num = int(filter_num/2)


  # Decode
  for i in range(num_layers-1):
      skip_num = num_layers-i-2
      block_out = decode_block(filter_num, block_in, skip_layers[skip_num],num_layers-i-1,norm)
      filter_num = int(filter_num/2)
      block_in = block_out

  model_out = keras.layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(block_out)

  model = keras.Model(Unet_input, model_out, name="UNet")
  return model

#-------------------------------------------------------------------------
# Functions to fit, evaluate, and test the model
#-------------------------------------------------------------------------

def fit_n_assess_model(nmodel, model_info, data, plot_filepath):

    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    start = datetime.now()

    # Compile model
    print("- Compile model ",model_info['name'])
    nmodel.compile(loss=model_info['loss'], optimizer=model_info['optimizer'], metrics=model_info['metrics'])

    
    # Fit the model with the training data from the data dictionary.
    print("- Fit model ",model_info['name'])
    nhist1 = nmodel.fit(x = X_train, y = y_train, epochs=model_info['epochs'],
                        callbacks=[TimeCallback()],
                        validation_split=0.2,)
    # Collect fit stats
    stop = datetime.now()
    model_info['tdelta'] = str(stop-start)
    print("Runtime: ", str(stop-start))

    print("- Evaluate model ",model_info['name'])
    model_info['test_accuracy'] = str(round(nmodel.evaluate(X_test, y_test, verbose=True)[1],4))
    print("Test accuracy: ",model_info['test_accuracy'])

    print(nhist1.history)
    plot_model_loss(model_info, nhist1.history, plot_filepath, title=nmodel.name)
    print("Loss and accuracy plots saved.")
    return nhist1

#-------------------------------------------------------------------------
# Plotting functions
#-------------------------------------------------------------------------

def plot_model_loss(model_info, hist, pict_filepath, title="model loss"):
    # This function creates the train & validation loss comparison plots and saves them to a file
    #
    # Output: None

    num_metrics = int(len(hist.keys()))
    half_num_metrics = int(num_metrics/2)

    fig = plt.figure(figsize=(10, 8))

    for key_num in range(half_num_metrics):
      train_key = list(hist.keys())[key_num]
      val_key = list(hist.keys())[key_num+half_num_metrics]

      train_data = hist[train_key]
      val_data = hist[val_key]
      
      ax = fig.add_subplot(half_num_metrics,1,key_num+1)  
      ax.plot(range(0,model_info['epochs']), train_data, label="training set")
      ax.plot(range(0,model_info['epochs']), val_data, label="validation set")
      
      if train_key == 'loss':
        ax.set_ylabel(model_info['loss'])
      else:
        ax.set_ylabel(train_key)#.name)
      ax.legend(loc="upper right")

    plot_title = title+" \n("
    cnt = 0
    for key, value in model_info.items():
      if key != 'loss':
        if key == "optimizer":
          if type(value) != str:
            value = value.name
        plot_title += key + ": "+ str(value) + ", "
        cnt += 1
        if cnt == 4:
          plot_title += "\n"
          cnt = 0
    plot_title += ")"

    fig.suptitle(plot_title, y=1.02, fontsize=12)
    plt.tight_layout()
    #plt.show()

    filename = pict_filepath + title+"_loss_metrics.png"
    fig.savefig(filename, bbox_inches='tight')
    plt.clf()

def plot_images(X,y,y_pred,pict_filepath,y_label=None):
  rows = 1
  cols = 2+len(y_pred)
  fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
  axes = axes.flatten()
  
  ax = axes[0]
  ax.imshow(X)
  ax.text(0.5, 0,'image',fontsize=15,horizontalalignment='center', verticalalignment='top',transform=ax.transAxes)
  ax.axis('off')

  ax = axes[1]
  ax.imshow(y,cmap='grey')
  ax.text(0.5, 0,'mask',fontsize=15,horizontalalignment='center', verticalalignment='top',transform=ax.transAxes)
  ax.axis('off')

  for i in range(len(y_pred)):
    ax = axes[i+2]
    ax.imshow(y_pred[i],cmap='grey')
    if y_label:
      ax.text(0.5, 0,y_label[i],fontsize=15,horizontalalignment='center', verticalalignment='top',transform=ax.transAxes)
    else:
      ax.text(0.5, 0,'model prediction',fontsize=15,horizontalalignment='center', verticalalignment='top',transform=ax.transAxes)
    ax.axis('off')

    filename = pict_filepath+"Results_"+y_label[i]+".png"
    fig.savefig(filename, bbox_inches='tight')
    plt.clf()

    print("Results image saved")

#-------------------------------------------------------------------------------------------