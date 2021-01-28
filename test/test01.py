from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
filepath='../data/modelcheckpoint/'
filename='_{epoch:02d}-{val_loss:.4f}.hdf5'
modelpath = "".join([filepath, "k45_", '{timer}', filename])

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.distribute import distributed_file_utils
@keras_export('keras.callbacks.ModelCheckpoint')
class MyModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def _get_file_path(self, epoch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        try:
        # `filepath` may contain placeholders such as `{epoch:02d}` and
        # `{mape:.2f}`. A mismatch between logged metrics and the path's
        # placeholders can cause formatting to fail.
            file_path = self.filepath.format(epoch=epoch + 1, timer=datetime.datetime.now().strftime('%m%d_%H%M'), **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath: "{}". '
                        'Reason: {}'.format(self.filepath, e))
        self._write_filepath = distributed_file_utils.write_filepath(
            file_path, self.model.distribute_strategy)
        return self._write_filepath