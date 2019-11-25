# AI Programming with Python

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

**Part 1 - Developing an Image Classifier with Deep Learning**

In this first part of the project, you'll work through a Jupyter notebook to implement an image classifier with PyTorch. We'll provide some tips and guide you, but for the most part the code is left up to you. As you work through this project, please refer to the rubric for guidance towards a successful submission.


This notebook will be required as part of the project submission. After you finish it, make sure you download it as an HTML file and include it with the files you write in the next part of the project.

We've provided you a workspace with a GPU for working on this project. If you'd instead prefer to work on your local machine, you can find the files on GitHub here.

If you are using the workspace, be aware that saving large files can create issues with backing up your work. You'll be saving a model checkpoint in Part 1 of this project which can be multiple GBs in size if you use a large classifier network. Dense networks can get large very fast since you are creating N x M weight matrices for each new layer. In general, it's better to avoid wide layers and instead use more hidden layers, this will save a lot of space. Keep an eye on the size of the checkpoint you create. You can open a terminal and enter ls -lh to see the sizes of the files. If your checkpoint is greater than 1 GB, reduce the size of your classifier network and re-save the checkpoint.

**Part 2 - Building the command line application**

Now that you've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. Your application should be a pair of Python scripts that run from the command line. For testing, you should use the checkpoint you saved in the first part.

Specifications

The project submission must include at least two files train.py and predict.py. The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Our suggestion is to create a file just for functions and classes relating to the model and another one for utility functions like loading data and preprocessing images. Make sure to include all files necessary to run train.py and predict.py in your submission.
    

Part 2 - Command Line Application
Criteria 	Specification
Training a network 	train.py successfully trains a new network on a dataset of images
Training validation log 	The training loss, validation loss, and validation accuracy are printed out as a network trains
Model architecture 	The training script allows users to choose from at least two different architectures available from torchvision.models
Model hyperparameters 	The training script allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
Training with GPU 	The training script allows users to choose training the model on a GPU
Predicting classes 	The predict.py script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability
Top K classes 	The predict.py script allows users to print out the top K classes along with associated probabilities
Displaying class names 	The predict.py script allows users to load a JSON file that maps the class values to other category names
Predicting with GPU 	The predict.py script allows users to use the GPU to calculate the predictions

## For training.py (data directory: flowers)

**Basic usage:**

`<python train.py data_directory>`

Example:
`<python train.py flowers>`

**Set directory to save checkpoints:**

`<python train.py data_dir --save_dir save_directory>`

Example:

`<python train.py flowers --save_dir some_dir --gpu>`  (--gpu for activating Cuda)

**Choose architecture, (options are: vgg16 or densenet161):**

`<python train.py data_dir --arch "vgg16">`

Example:

`<python train.py flowers --arch "vgg16" --gpu>` (--gpu for activating Cuda)

**Set hyperparameters:**

`<python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20>`

Example:

`<python train.py flowers --learning_rate 0.01 --hidden_units 512 --epochs 20>`

**Use GPU for training:**

`<python train.py data_dir --gpu>`

Example:

`<python train.py flowers --gpu>`

## for predicting a image category


**Basic usage:**

`<python predict.py /path/to/image checkpoint>`

**Return top KKK most likely classes:**

`<python predict.py input checkpoint --top_k 3>`

Example:

`<python predict.py flowers/test/16/image_06657.jpg some_dir/test.pth --top_k 3>`

**Use a mapping of categories to real names:**

`<python predict.py input checkpoint --category_names cat_to_name.json>`

Example:

`<python predict.py flowers/test/16/image_06657.jpg  some_dir/test.pth --top_k 3 --category_names cat_to_name.json>`

**Use GPU for inference:**

`<python predict.py input checkpoint --gpu>`

Example:

`<python predict.py flowers/test/16/image_06657.jpg  some_dir/test.pth --gpu>`


