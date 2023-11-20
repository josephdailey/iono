"""igs_tec_slices dataset."""

from gc import disable
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

def _rectify_int_keys(ds: tf.data.Dataset, key: str):
  # This step is needed because
  # (FOR _SOME_ REASON)
  # TFDS sorts negative int keys after positive ones,
  # and provides no built-in sort method.
  # This does NOT sort the data, just puts - before +.
  # Make sure the input is not shuffled!!!!
  neg_keys = ds.filter(lambda x: x[key] < 0)
  nonneg_keys = ds.filter(lambda x: x[key] >= 0)
  return neg_keys.concatenate(nonneg_keys)

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for igs_tec_slices dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'timecode': np.int32,
            'prev_maps': tfds.features.Tensor(shape=(12, 71, 73, 1), dtype=np.float32),
            'next_map': tfds.features.Tensor(shape=(71, 73, 1), dtype=np.float32),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('prev_maps', 'next_map'),  # Set to `None` to disable
        homepage='https://github.com/josephdailey/iono',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # Downloads the data and defines the splits
    ds = tfds.load("igs_tec", split="all")
    ds = _rectify_int_keys(ds, "timecode")

    # Print the first and last few keys just to make sure the sort worked
    for x in ds.take(10):
      print(x["timecode"].numpy())

    # Returns the Dict[split names, Iterator[Key, Example]]
    return {'_all': self._generate_examples(ds)}

  def _generate_examples(self, parent_ds):
    """Yields examples."""
    # TODO: Implement variable windows?
    # Ideally a shift that is coprime with the samples-per-day is chosen,
    # so that we eventually get windows covering all times of day.
    # A shift of 1 creates every possible window,
    # but this visits each map 12 times and I am afraid of overfitting.
    inputs = parent_ds.window(12, shift = 11, drop_remainder=True)
    targets = parent_ds.skip(12).window(1, shift = 11)
    pairs = tf.data.Dataset.zip(inputs, targets)
    for pair in pairs:
      timecode = pair[1]["timecode"].get_single_element().numpy()
      yield int(timecode), {
          'timecode': timecode,
          'prev_maps': np.asarray([map for map in pair[0]["tec_map"]]),
          'next_map': pair[1]["tec_map"].get_single_element().numpy()
      }
