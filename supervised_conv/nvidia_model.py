from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from keras.optimizers import Adam


class NvidiaModel:
    @staticmethod
    def build_model(input_shape, learning_rate=1.0e-4, keep_prob=0.5):
        """
        NVIDIA model used
        Image normalization to avoid saturation and make gradients work better.
        Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
        Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
        Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
        Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
        Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
        Drop out (0.5)
        Fully connected: neurons: 100, activation: ELU
        Fully connected: neurons: 50, activation: ELU
        Fully connected: neurons: 10, activation: ELU
        Fully connected: neurons: 1 (output)
        """
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=input_shape))
        model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(64, 3, 3, activation='elu'))
        model.add(Conv2D(64, 3, 3, activation='elu'))
        model.add(Dropout(keep_prob))
        model.add(Flatten())
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        # model.add(Dense(1))
        model.add(Dense(2))
        model.summary()

        model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

        return model