# Run HW3 Retina Scan with U-model: tf_dl_hw3

tf_dl_hw3 runs all of the models needed for Deep Learning HW3.

## Usage

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


