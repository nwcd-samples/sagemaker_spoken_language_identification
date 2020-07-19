from constants import *

import time
import re
import sys
import os

sys.path.append("./")
os.system("pip install -i https://mirrors.163.com/pypi/simple/ imageio")
os.system("pip install -i https://mirrors.163.com/pypi/simple/ speechpy")

#os.system("pip install matplotlib")
#os.system("pip install easydict")
#os.system("pip install glog")
#os.system("pip install speechpy")


print(sys.path)



# supress all warnings (especially matplotlib warnings)
import warnings
warnings.filterwarnings("ignore")

# RANDOMNESS
# https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development

os.environ['PYTHONHASHSEED'] = '0'

import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

# supress tensorflow debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# disable auto tune
# https://github.com/tensorflow/tensorflow/issues/5048
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

import tensorflow as tf
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(SEED)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from sklearn import preprocessing
from sklearn.metrics import classification_report

from keras.models import Model, load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten
from keras.layers import Dropout, Input, Activation
from keras.optimizers import Nadam, SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras import regularizers

import folds

import common


def build_model(input_shape, language_list):
        
    model = Sequential()

    # 40x1000

    model.add(Conv2D(
        16,
        (3, 3),
        strides=(1, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(0.001),#0.001
        input_shape=input_shape))
    model.add(Activation('elu'))
    # add by shishuai
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # 20x500

    model.add(Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(0.001)))#0.001
    model.add(Activation('elu'))
        # add by shishuai
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # 10x250

    model.add(Conv2D(
        64,
        (3, 3),
        strides=(1, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(0.001)))#0.001
    model.add(Activation('elu'))
        # add by shishuai
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # 5x125

    model.add(Conv2D(
        128,
        (3, 5),
        strides=(1, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(0.001)))#0.001
    model.add(Activation('elu'))
        # add by shishuai
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 5), strides=(1, 5), padding='same'))

    # 5x25

    model.add(Conv2D(
        256,
        (3, 5),
        strides=(1, 1),
        padding='same',
        kernel_regularizer=regularizers.l2(0.001)))#0.001
    model.add(Activation('elu'))
        # add by shishuai
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 5), strides=(1, 5), padding='same'))
    model.add(AveragePooling2D(
        pool_size=(5, 5),
        strides=(5, 5),
        padding='valid'))

    # 1x1

    model.add(Flatten())

    model.add(Dense(
        32,
        activation='elu',
        kernel_regularizer=regularizers.l2(0.001)))#0.001

    model.add(Dropout(0.5))

    model.add(Dense(len(language_list)))
    model.add(Activation('softmax'))

#     sgd = SGD(lr=args.learning_rate, decay=args.decay, momentum=args.momentum, nesterov=False)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    model.compile(
        loss='categorical_crossentropy',
#         optimizer=sgd,
        optimizer=adam,
        metrics=['accuracy'])

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument(
        '--test',
        dest='test',
        action='store_true',
        help='test the previously trained model against the test set')
    
    parser.add_argument(
        "-e",
        "--epoch_count",
        type=int,
        nargs="?",
        help="Epoch count",
        default=100,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="batch size",
        default=8,
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        help="Directory containing data",
        default="/opt/ml/input/data/train",
        
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="output data dir",
        default="/opt/ml/model/",
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        type=str,
        help="SageMaker will give you this parameter no mater you have specified or not.",
        default=1,
    )
    parser.add_argument(
        "-c",
        "--create_data",
        type=int,
        help="提取特征进行合并， 0: 不提取; 1: 提取",
        default=0,
    )

    
    parser.set_defaults(test=False)

    args = parser.parse_args()

    input_shape = (FB_HEIGHT, WIDTH, COLOR_DEPTH)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(" output_dir :  ", args.output_dir)
        
    if args.create_data == 1:
        print('重新生成训练数据')
        folds.main(args.input_dir)
    else:
        print('训练数据已经生成')
    # 获取 语言分类信息 language_label.txt；
    temp_build_fold = args.input_dir

    language_list = common.load_language_map(temp_build_fold)
    
    if args.test:
        model = load_model(os.path.join(args.output_dir, 'model.h5'))

        input_shape = (FB_HEIGHT, WIDTH, COLOR_DEPTH)
        label_binarizer, clazzes = common.build_label_binarizer()

        test_labels, test_features, test_metadata = common.load_data(
            label_binarizer, temp_build_fold, 'test', input_shape)

        common.test(test_labels, test_features, test_metadata, model, clazzes)
    else:
        accuracies = []
        generator = common.train_generator(temp_build_fold, input_shape)

        first = True
        for (train_labels,
             train_features,
             test_labels,
             test_features,
             test_metadata,
             clazzes) in generator:

            # TODO reset tensorflow

            model = build_model(input_shape, language_list)
            if first:
                model.summary()
                first = False

            checkpoint = ModelCheckpoint(
                os.path.join(args.output_dir, 'model.h5'),
                monitor='val_loss',
                verbose=0,
                save_best_only=True,
                mode='min')

            earlystop = EarlyStopping(
                monitor='val_loss',
                min_delta=0,
                patience=7,
                verbose=0,
                restore_best_weights = True,
                mode='auto')

            model.fit(
                train_features,
                train_labels,
                epochs=args.epoch_count,
                callbacks=[checkpoint, earlystop],
                verbose=2, #日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
                validation_data=(test_features, test_labels),
                batch_size=args.batch_size)
                      
#             model = load_model(os.path.join(args.output_dir, 'model.h5'))

#             model2 = tf.keras.models.load_model(os.path.join(args.output_dir, 'model.h5'))
#             model2.save(os.path.join(args.output_dir, '1'), save_format='tf')  # 导出tf格式的模型文件  save_model/1
            
            scores = model.evaluate(test_features, test_labels, verbose=0)
            accuracy = scores[1]

            print('Accuracy:', accuracy)
            accuracies.append(accuracy)



            common.test(
                test_labels,
                test_features,
                test_metadata,
                model,
                clazzes)

        accuracies = np.array(accuracies)
        # add by shishuai, delete model.h5, only save tf .pb files.
#         os.remove('/opt/ml/model/model.h5')
        
        print('\n## Summary\n')
        print("Mean: {mean}, Std {std}".format(
            mean=accuracies.mean(),
            std=accuracies.std()))
