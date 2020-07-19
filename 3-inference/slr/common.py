from constants import *
import os
import numpy as np
import pandas as pd
import glob
from sklearn import preprocessing
from sklearn.metrics import classification_report


def can_ignore(file, key):
    if key in file:
        return True
    return False


def flatten(binary_labels):
    return np.argmax(binary_labels, axis=1)


def test(labels, features, metadata, model, clazzes, title="test"):
    probabilities = model.predict(features, verbose=0)

    expected = flatten(labels)
    actual = flatten(probabilities)

    print("\n## {title}\n".format(title=title))

    max_probabilities = np.amax(probabilities, axis=1)

    print("Average confidence: {average}\n".format(
        average=np.mean(max_probabilities)))

    errors = 0
    for label, pred in zip(expected, actual):
        if label != pred:
            errors += 1
        #print('label: {} \t pred: {} '.format( label, pred))


    print("Amount of errors: {}  total: {}".format(errors, len(expected)))


def load_data(label_binarizer, input_dir, group, input_shape):
    all_metadata = []
    all_features = []

    metadata_file_list = glob.glob(os.path.join(input_dir, "{group}_metadata.fold_*.npy".format(group=group)))
    metadata_file_list = sorted(metadata_file_list)
    data_file_list = glob.glob(os.path.join(input_dir, "{group}_data.fold_*.npy".format(group=group)))
    data_file_list = sorted(data_file_list)

    # enumerate(glob.glob():
    for metadata_file, data_file in zip(metadata_file_list, data_file_list):
        metadata = np.load(metadata_file)

        features = np.memmap(
            data_file,
            dtype=DATA_TYPE,
            mode='r',
            shape=(len(metadata),) + input_shape)

        all_metadata.append(metadata)
        all_features.append(features)

    all_metadata = np.concatenate(all_metadata)
    all_features = np.concatenate(all_features)
    all_labels = label_binarizer.transform(all_metadata[:, 0])

    print("[{group}] labels: {labels}, features: {features}".format(
        group=group, labels=all_labels.shape, features=all_features.shape))

    return all_labels, all_features, all_metadata


def load_language_map(input_dir):
    language_label_file = os.path.join(input_dir, 'language_label.txt')
    assert os.path.exists(language_label_file), '{} 文件不存在'.format(language_label_file)
    with open(language_label_file, 'r') as f:
        line = f.readline()
    return line.split(',')




def build_label_binarizer(input_dir):
    label_binarizer = preprocessing.LabelBinarizer()
    label_binarizer.fit(load_language_map(input_dir))
    clazzes = list(label_binarizer.classes_)
    print("Classes:", clazzes)

    return label_binarizer, clazzes


def train_generator(input_dir, input_shape):
    label_binarizer, clazzes = build_label_binarizer(input_dir)

    train_labels, train_features, train_metadata = load_data(
        label_binarizer,
        input_dir,
        'train',
        input_shape)

    test_labels, test_features, test_metadata = load_data(
        label_binarizer,
        input_dir,
        'test',
        input_shape)

    yield (train_labels, train_features, test_labels,
           test_features, test_metadata, clazzes)

    del train_labels
    del train_features
    del train_metadata

    del test_labels
    del test_features
    del test_metadata



def remove_extension(file):
    return os.path.splitext(file)[0]


def get_filename(file):
    return os.path.basename(remove_extension(file))



if __name__ == "__main__":
    generator = train_generator('../build/folds', (FB_HEIGHT, WIDTH, COLOR_DEPTH))
    for train_labels, train_features, test_labels, test_features, test_metadata, clazzes in generator:
        print('train_labels: ', train_labels.shape)
    # input_shape = (FB_HEIGHT, WIDTH, COLOR_DEPTH)
    # label_binarizer, clazzes = build_label_binarizer()
    #
    # test_labels, test_features, test_metadata = load_data(
    #     label_binarizer, '../build/folds', 'train', input_shape)