# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## For training.py (data directory: flowers)

Basic usage:

`<python train.py data_directory>`

Example:
`<python train.py flowers>`

Set directory to save checkpoints:

`<python train.py data_dir --save_dir save_directory>`

Example:

`<python train.py flowers --save_dir some_dir --gpu>`  (--gpu for activating Cuda)

Choose architecture, (options are: vgg16 or densenet161):

`<python train.py data_dir --arch "vgg16">`

Example:

`<python train.py flowers --arch "vgg16" --gpu>` (--gpu for activating Cuda)

Set hyperparameters:

`<python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20>`

Example:

`<python train.py flowers --learning_rate 0.01 --hidden_units 512 --epochs 20>`

Use GPU for training:

`<python train.py data_dir --gpu>`

`<python train.py flowers --gpu>`

## for predicting a image category

Basic usage:

`<python predict.py /path/to/image checkpoint>`

Return top KKK most likely classes:

`<python predict.py input checkpoint --top_k 3>`

Use a mapping of categories to real names:

`<python predict.py input checkpoint --category_names cat_to_name.json>`

Example:

`<python predict.py flowers/test/16/image_06657.jpg  some_dir/test.pth --top_k 3 --category_names cat_to_name.json>`

Use GPU for inference:

`<python predict.py input checkpoint --gpu>`
