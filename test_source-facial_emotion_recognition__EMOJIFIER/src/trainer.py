import os
import sys
import cv2
import time
import logging
import json
import tensorflow as tf
import numpy as np
import glob
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from src.data_manager import EmojifierDataManager
from src.__init__ import *

logger = logging.getLogger('emojifier.model')


def weight_variable(shape):
    initializer = tf.keras.initializers.TruncatedNormal(stddev=0.1)
    return tf.Variable(initializer(shape), trainable=True)


def bias_variable(shape):
    initializer = tf.constant(0.1, shape=shape)
    return tf.Variable(initializer, trainable=True)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(tf.nn.batch_normalization(conv2d(input, W) + b, 0, 1, 0, 1, 1e-5))


def full_layer(input, size):
    insize = int(input.shape[1])
    W = weight_variable([insize, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b


def model(x, keep_prob):
    C1, C2, C3 = 30, 50, 80
    F1 = 512

    conv1_1 = conv_layer(x, shape=[3, 3, 1, C1])
    conv1_1_pool = max_pool_2x2(conv1_1)

    conv1_2 = conv_layer(conv1_1_pool, shape=[3, 3, C1, C2])
    conv1_2_pool = max_pool_2x2(conv1_2)

    conv1_drop = tf.nn.dropout(conv1_2_pool, rate=1 - keep_prob)

    conv2_1 = conv_layer(conv1_drop, shape=[3, 3, C2, C3])
    conv2_1_pool = max_pool_2x2(conv2_1)

    conv2_flat = tf.reshape(conv2_1_pool, [-1, 6 * 6 * C3])
    conv2_drop = tf.nn.dropout(conv2_flat, rate=1 - keep_prob)

    full1 = tf.nn.relu(full_layer(conv2_drop, F1))
    full1_drop = tf.nn.dropout(full1, rate=1 - keep_prob)

    y_conv = full_layer(full1_drop, len(EMOTION_MAP))
    return y_conv


def test(emoji_data, model):
    logger.info('CALCULATING TESTSET ACCURACY ...')
    L = len(emoji_data.test.labels)

    x = emoji_data.test.images
    y = emoji_data.test.labels

    accs = []

    for i in tqdm.tqdm(range(0, L, 30)):
        x_i = x[i:i+30].reshape(-1, 48, 48, 1)
        y_i = y[i:i+30]

        acc = model.evaluate(x_i, y_i, verbose=0)[1]
        accs.append(acc)

    avg_acc = np.mean(accs)
    logger.critical(f'test-accuracy: {avg_acc * 100:.2f}%')


if __name__ == '__main__':
    CHECKPOINT_SAVE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, 'model_checkpoints')

    if not os.path.exists(CHECKPOINT_SAVE_PATH):
        os.makedirs(CHECKPOINT_SAVE_PATH)

    BATCH_SIZE = config_parser.getint('MODEL_HYPER_PARAMETERS', 'batch_size')
    STEPS = config_parser.getint('MODEL_HYPER_PARAMETERS', 'train_steps')
    LEARNING_RATE = config_parser.getfloat('MODEL_HYPER_PARAMETERS', 'learning_rate')
    KEEP_PROB = config_parser.getfloat('MODEL_HYPER_PARAMETERS', 'dropout_keep_prob')

    emoset = EmojifierDataManager()

    logger.info(f"Number of train images: {len(emoset.train.images)}")
    logger.info(f"Number of train labels: {len(emoset.train.labels)}")
    logger.info(f"Number of test images: {len(emoset.test.images)}")
    logger.info(f"Number of test labels: {len(emoset.test.labels)}")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(48, 48, 1)),
        tf.keras.layers.Conv2D(30, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(50, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(1 - KEEP_PROB),
        tf.keras.layers.Conv2D(80, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(1 - KEEP_PROB),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(1 - KEEP_PROB),
        tf.keras.layers.Dense(len(EMOTION_MAP), activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(emoset.train.images, emoset.train.labels, epochs=STEPS, batch_size=BATCH_SIZE)

    test(emoset, model)

    model.save(os.path.join(CHECKPOINT_SAVE_PATH, 'model.h5'))
    logger.info(f"Model saved in path: {CHECKPOINT_SAVE_PATH}")

