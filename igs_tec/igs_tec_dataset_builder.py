"""igs_tec dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
from gnssanalysis.gn_io.ionex import *

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for igs_tec dataset."""

  VERSION = tfds.core.Version('1.0.1')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.0.1': 'Disabled shuffling (will need postprocessing)'
  }

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Place `igs_data.tar.gz` in `manual_dir`.
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'timecode': np.int32,
            'tec_map': tfds.features.Tensor(shape=(71, 73, 1), dtype=np.float32),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=(None),  # Set to `None` to disable
        homepage='https://github.com/josephdailey/iono',
        disable_shuffling = True
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    archive_path = dl_manager.manual_dir / "igs_data.tar.gz"
    extracted_path = dl_manager.extract(archive_path)

    return {'_all':self._generate_examples(extracted_path)}

  def _generate_examples(self, path):
    """Yields examples."""
    keys = []
    for file in os.scandir(path / "data"):
      raw = read_ionex(file.path)
      raw = raw[~raw.index.duplicated(keep='last')]
      raw_bytime = raw.groupby(level="DateTime")
      for timecode in raw_bytime.groups:
        # Guard against duplicate timestamps, lest TFDS get grumpy about collisions;
        # Technically this check is O(n^2), but I am lazy and this script should only run once.
        if timecode not in keys:
          keys.append(timecode)
          ionex_array = raw_bytime.get_group(timecode).xs("TEC", level="Type").to_numpy(dtype=np.float32)
          ionex_array = np.expand_dims(ionex_array, axis=-1)
          yield int(timecode), {
            'timecode': timecode,
            'tec_map': ionex_array
          }
