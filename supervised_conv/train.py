import os
import datetime
import argparse

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from common.image_processor import batch_generator, INPUT_SHAPE
from common.utils import s2b

from supervised_conv.nvidia_model import NvidiaModel

np.random.seed(0)


def load_data(args):
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    data_df['throttle'] -= data_df['reverse']

    X = data_df[['center', 'left', 'right']].values
    y = data_df[['steering', 'throttle']].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size)

    print('Dataset size: {}', X_train.shape)

    return X_train, X_valid, y_train, y_valid


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    # mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc, 
    # this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint('model-{epoch:03d}-nvidia-2-outs-old-dataset.h5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)


def build_model(model_name):
    if model_name == 'nvidia':
        return NvidiaModel.build_model(INPUT_SHAPE)
    else:
        raise ValueError("Unknown model name")


def main():
    parser = argparse.ArgumentParser(description='Self driving car')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='data')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.20)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.4)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=50)
    parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=20000)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=50)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    parser.add_argument('-m', help='use model. Choose from \{nvidia\}', dest='model', type=str, default='nvidia')
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args.model) 
    train_model(model, args, *data)


if __name__ == '__main__':
    main()
