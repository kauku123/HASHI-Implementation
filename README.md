# HASHI-Implementation

## Python implementation of HASHI paper


To use this implementation, the following depencies are needed:


* Python 2.7
* Numpy
* scipy
* PyTorch 1.0
* CUDA tookit 9 (for GPU computation)
* PIL (Python Image Library)

The implementation runs as a single Python file and takes as command line arguments the following:

* Path to the directory containing the WSI’s
* Path to the directory of the the extracted tiles
* Path to the text file to which softmax probabilities of the sampled tiles are written.


To run:
**python full_hashi_algo_iter.py  path_to_wsi_dir  path_to_sampled_tiles  path_to_softmax_textfile**

Place the trained CNN model to be used in HASHI scheme in the **models_cnn** directory.

Currently, CNN models built in PyTorch are supported.
