# Run HW1 Deep Learning Models: tf_dl_hw1

tf_dl_hw1 runs all of the models needed for Deep Learning HW1.

## Usage

The model to run (input number of corresponding model):   
	1:linear regression,  
	2:DNN-16,  
	3:DNN-30-8,  
	4:DNN-30-16-8,  
	5:DNN-30-16-8-4
	
The learning rate has a default value of 0.01.
The epoch number has a default value of 11.

```python

# runs linear regression model
py trf_dl_hw1 -f cancer_reg.csv -m 1

# runs DNN-16
py trf_dl_hw1 -f cancer_reg.csv -m 2

# runs DNN-30-8
py trf_dl_hw1 -f cancer_reg.csv -m 3

# runs DNN-30-16-8
py trf_dl_hw1 -f cancer_reg.csv -m 4

# runs DNN-30-16-8-4
py trf_dl_hw1 -f cancer_reg.csv -m 5

# runs DNN-16 with custom learning rate and epoch number
py trf_dl_hw1 -f cancer_reg.csv -m 2 -l 0.001 -e 20

```