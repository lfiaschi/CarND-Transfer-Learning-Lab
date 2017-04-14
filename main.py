import pickle
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)
    
    X_train=X_train.squeeze()
    y_train=y_train.squeeze()
    X_val=X_val.squeeze()
    y_val=y_val.squeeze()
    
    num_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]
    
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    dropout = .3
    batch_size = 128
    epochs = 500
    
    model = Sequential()
    model.add(Dense(256, activation='relu',input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])


    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=30)

    model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=.3, 
          callbacks=[early_stopping])
    
    score = model.evaluate(X_val, y_val, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    import gc; gc.collect()
    
    
# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
