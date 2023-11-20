import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import datasets.igs_tec_slices
import numpy as np
import matplotlib.pyplot as plt

def get_ds_stats(ds):
  maps = np.asarray([batch[1].numpy() for batch in ds])
  return maps.mean(), maps.var()

mean, var = get_ds_stats(tfds.load("igs_tec_slices", split="_all", as_supervised=True))
print(mean, var)

train_ds = tfds.load("igs_tec_slices", split="_all[:70%]", as_supervised=True, data_dir="datasets", shuffle_files=True).shuffle(5000).batch(10)
val_ds = tfds.load("igs_tec_slices", split = "_all[70%:90%]", as_supervised=True, data_dir="datasets", shuffle_files=True).shuffle(5000).batch(10)
test_ds = tfds.load("igs_tec_slices", split = "_all[90%:]", as_supervised=True, data_dir="datasets", shuffle_files=True).shuffle(5000).batch(10)

forecaster = tf.keras.models.Sequential([
            layers.InputLayer(input_shape=(12, 71, 73, 1)),
            layers.Normalization(axis=None, mean=mean, variance=var),
            layers.ConvLSTM2D(filters=64, kernel_size=(5, 5), padding='same', return_sequences=True),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True),
            layers.BatchNormalization(),
            layers.ConvLSTM2D(filters=64, kernel_size=(1, 1), padding='same'),
            layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same')])

forecaster.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

history1 = forecaster.fit(train_ds, epochs=5, validation_data=val_ds)
forecaster.evaluate(test_ds)

for input, target in test_ds.take(5):
  plt.imshow(forecaster.predict(input)[0, :, :, 0])
  plt.show()
  plt.imshow(target[0, :, :, 0])
  plt.show()