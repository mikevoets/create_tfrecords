from absl import app
import random
import tensorflow as tf
from dataset_utils import (
    _dataset_exists,
    _get_filenames_and_classes,
    write_label_file,
    _convert_dataset,
)


from absl import flags

flags.DEFINE_string("dataset_dir", "../data", "String: Your dataset directory")
flags.DEFINE_float(
    "validation_size",
    0.3,
    "Float: The proportion of examples in the dataset to be used for validation",
)
flags.DEFINE_integer(
    "num_shards", 2, "Int: Number of shards to split the TFRecord files"
)
flags.DEFINE_integer("random_seed", 0, "Int: Random seed to use for repeatability.")
flags.DEFINE_string(
    "tfrecord_filename",
    "outputtfrecord",
    "String: The output filename to name your TFRecord file",
)
FLAGS = flags.FLAGS


def main(argv):

    if not FLAGS.tfrecord_filename:
        raise ValueError(
            "tfrecord_filename is empty. Please state a tfrecord_filename argument."
        )

    if not FLAGS.dataset_dir:
        raise ValueError("dataset_dir is empty. Please state a dataset_dir argument.")

    if _dataset_exists(
        dataset_dir=FLAGS.dataset_dir,
        _NUM_SHARDS=FLAGS.num_shards,
        output_filename=FLAGS.tfrecord_filename,
    ):
        return None
    photo_filenames, class_names = _get_filenames_and_classes(FLAGS.dataset_dir)

    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    num_validation = int(FLAGS.validation_size * len(photo_filenames))
    random.seed(FLAGS.random_seed)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]

    _convert_dataset(
        "train",
        training_filenames,
        class_names_to_ids,
        dataset_dir=FLAGS.dataset_dir,
        tfrecord_filename=FLAGS.tfrecord_filename,
        _NUM_SHARDS=FLAGS.num_shards,
    )
    _convert_dataset(
        "validation",
        validation_filenames,
        class_names_to_ids,
        dataset_dir=FLAGS.dataset_dir,
        tfrecord_filename=FLAGS.tfrecord_filename,
        _NUM_SHARDS=FLAGS.num_shards,
    )

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, FLAGS.dataset_dir)


if __name__ == "__main__":
    app.run(main)
