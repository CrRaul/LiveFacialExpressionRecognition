

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

def load_model(weights_path=None, shape=(128, 128,1)):

    print(shape)
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, \
            padding='same', activation='relu', \
            input_shape=(128,128,1)))
    # 128*128*64
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    # 14*14*64

    model.add(Conv2D(filters=64, kernel_size=3, strides=2, \
            padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=2, \
            padding='same', activation='relu'))
    # 14*14*128
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    # 7*7*128

    model.add(Conv2D(filters=128, kernel_size=3, strides=2  , \
            padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=3, strides=2  , \
            padding='same', activation='relu'))
    # 7*7*256
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    # 4*4*256

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    print ("Created model successfully")
    if weights_path:
        model.load_weights(weights_path)

    return model
