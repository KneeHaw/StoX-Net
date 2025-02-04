# The following will detail structure and dependencies for the MIT toy model implemented stochastic ADC conversion

- **_I recommend to run this code in an IDE (pycharm, vscode, etc.) as I have not tested its commandline arguments._**

## Some Dependencies:
1. TODO

*Note these dependencies will have their own dependencies, so just install anything that causes an error*

## Project structure:
### argparser.py
- This file contains a simple function that sets a large list of parser arguments. This is where many components of the
model can be changed.
  - Note that changing _bit-width_/_iterations_ in this file should have no effect as I have manually set it
  - You can change the _subarray_size_, _batch_size_, and other hyperparameters by altering the _'default'_ value
    - You can technically do this from terminal as arguments (hence args) but I usually do it in IDE side
- Set evaluate to "True" and resume to the saved model's absolute path in order to perform just inference

### trainer.py
- This is the main executable file that you will run!
- This contains the training loop and evaluation functions
- Note that on Line 15 you can select your CUDA device, if you only have one device select _"0"_
- Lines 18 and 19 in conjunction with argparser's _'save-dir'_ specify the save directory for logs and models
- Everything else shouldn't matter

### utilities.py
- Loads the MNIST dataset or raises an error if not MNIST

### toyModel.py
- TODO

### StochasticLibQuan.py
- This is the file that contains the meat of all performed stochastic operations
- _StoX_Linear_ is the Linear Function replacement with stochastic capabilities
  - To ensure functionality, this is kept at an 4-bit input and 8 iterations
  - We can argue that the bits are streamed, meaning only a 1-bit DAC is necessary
    - Temporal expansion rather than representational expansion
  - Maps the operation to sequential MVM of subarrays
- _StoX_Conv2D_ is the standard image convolution replacement
  - To ensure functionality, this is kept at a 4-bit input and 8 iterations
  - Follows logic of linear representation
  - Maps the operation to sequential MVM of subarrays
  
### WeightQuantization.py
- Contains quantization functions for weights, barebones 1-bit implementation

### InputQuantization.py
- Contains 1-bit and multi-bit quantization for inputs

## Notes
- TODO