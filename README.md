updated for tensorflow2

# create_tfrecords
A simpler way of preparing large-scale image dataset by generalizing functions from TensorFlow-slim.

This is a forked and modified repository. The original code can be found [here](https://github.com/kwotsin/create_tfrecords).

### Requirements
1. Python 3

### Usage

    $ python create_tfrecord.py --dataset_dir=/path/to/dataset/ --tfrecord_filename=dataset_name

    # Example: python create_tfrecord.py --dataset_dir=/path/to/flowers --tfrecord_filename=flowers
    # Note that the dataset_dir should be the folder that contains the root directory and not the root directory itself.

### Arguments

#### Required arguments:

- dataset_dir (string): The directory to your dataset that is arranged in a structured way where your subdirectories keep classes of your images.

For example:

    flowers/
        tulips/
            ....jpg
            ....jpg
            ....jpg
        sunflowers/
            ....jpg
        roses/
            ....jpg
        dandelion/
            ....jpg
        daisy/
            ....jpg
  
- tfrecord_filename (string): The output name of your TFRecord files.

#### Optional Arguments
- validation_size (float): The proportion of the dataset to be used for evaluation.

- num_shards (int): The number of shards to split your TFRecord files into.

- random_seed (int): The random seed number for repeatability.
