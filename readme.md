# The following will detail structure and dependencies for the MIT toy model implemented stochastic ADC conversion

- **_I recommend to run this code in an IDE (pycharm, vscode, etc.) as I have not tested its commandline arguments._**

## Some Dependencies:
1. torch
2. torchvision
3. dill
4. argparse
5. Whatever else raises an error :)

*Note these dependencies will have their own dependencies, so just install anything that causes an error :)*

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
- This file contains the model definition with an option between architectures 1-4 and StoX or FP behavior
- **Architecture 1**
  1. Reduce image size to (14, 14) through average pooling
  2. StoX/FP Conv1 (14, 14) to (10, 10)
  3. StoX/FP Conv2 (10, 10) to (7, 7)
  4. Reduce image size to (3, 3) through average pooling
  5. FP Classifier (channels -> 10)
- **Architecture 2**
  1. Reduce image size to (14, 14) through average pooling
  2. StoX/FP Conv1 (14, 14) to (10, 10)
  3. StoX/FP Conv2 (10, 10) to (7, 7)
  4. Reduce image size to (3, 3) through average pooling
  5. StoX/FP Linear (channels * 9 -> 144)
  6. FP Classifier (144 -> 10)
- **Architecture 3**
  1. Reduce image size to (14, 14) through average pooling
  2. StoX/FP Conv1
  3. StoX/FP Conv2
  4. StoX/FP Conv3
  5. Reduce image size to (3, 3) through average pooling
  6. FP Classifier (channels * 9 -> 10)
- **Architecture 4**
  1. Reduce image size to (14, 14) through average pooling
  2. StoX/FP Conv1
  3. StoX/FP Conv2
  4. StoX/FP Conv3
  5. Reduce image size to (3, 3) through average pooling
  6. StoX/FP Linear (channels * 9 -> 144)
  7. FP Classifier (144 -> 10)
- Other options
  - _self.print_MTJops_andExit_ = True if you want to print the MTJ ops of one image
    - Does not take into account _num_iterations_, though it can with a simple code change
  - _self.StoX_ = bool determines whether the StoX/FP layers are StoX/FP
- Notes
  - Architectures 3 and 4 see **LARGE** increases in MTJ operations--I believe we should limit ourselves
  to architectures 1 and 2

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

### MTJQuantization.py
- THIS FILE CONTAINS A TODO. Implement the MATLAB interface here!
  - You will retain the backward part of the function (it is a STE)
  - The forward will be replaced by MATLAB equivalent of MTJ sampling

## Notes
- The training time for a model is extremely short, so you can adjust some parameters and alter your save directories
to see what happens.
- I recommend architectures 1 or 2! 2 is probably the best with its tradeoffs and rewards
  - The MTJ operations will be approximately 9000 with architectures 1 and 2 not counting iterations
  - I have included a trained **Architecture 2** with _subarray_size = 256_
- Accuracy should be at leas 85% with chosen models, if it's below this value then don't use it!