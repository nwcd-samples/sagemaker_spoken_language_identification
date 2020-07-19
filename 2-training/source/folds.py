# import imageio
from glob import glob
import os
import numpy as np
from sklearn.utils import shuffle
import time
import speechpy

from constants import *
import common
import math
import shutil


def generate_fold(
        output_dir,
        group,
        input_shape,
        normalize,
        output_shape,
        fold_index,
        fold_files,
        language_set):

    fold_files = sorted(fold_files)
    fold_files = shuffle(fold_files, random_state=SEED)

    metadata = []

    # create a file array
    filename = "{group}_data.fold_{index}.npy".format(
        group=group, index=fold_index)
    features = np.memmap(
        os.path.join(output_dir, filename),
        dtype=DATA_TYPE,
        mode='w+',
        shape=(len(fold_files),) + output_shape)

    # append data to a file array
    # append metadata to an array
    for index, fold_file in enumerate(fold_files):
        #print('Group {}  index {}  file: {} '.format(group, index, fold_file))

        language_set.add(fold_file.split('_')[0].split('/')[-1])
        filename = common.get_filename(fold_file)
        language = filename.split('_')[0]

        data = np.load(fold_file)[DATA_KEY]
        assert data.shape == input_shape
        assert data.dtype == DATA_TYPE

        features[index] = normalize(data)
        metadata.append((language, filename))

    assert len(metadata) == len(fold_files)

    filename = "{group}_metadata.fold_{index}.npy".format(
        group=group,
        index=fold_index)
    print("\tout filename: ", filename)
    np.save(
        os.path.join(output_dir, filename),
        metadata)

    # flush changes to a disk
    features.flush()
    del features


def generate_folds(
        input_dir,
        input_ext,
        output_dir,
        group,
        input_shape,
        normalize,
        split_count,
        output_shape,
        language_set):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_dir = os.path.join(input_dir, group)

    fold_files = glob(os.path.join(input_dir, '*' + input_ext))
    print("{}  files length: {}  ".format(group,  len(fold_files)))

    group_count = math.ceil(len(fold_files) / split_count)

    for i in range(split_count):


        generate_fold(
            output_dir,
            group,
            input_shape,
            normalize,
            output_shape,
            i,
            fold_files[i * group_count: (i+1)*group_count],
            language_set)



def normalize_fb(spectrogram):

    # Mean and Variance Normalization
    spectrogram = speechpy.processing.cmvn(
        spectrogram,
        variance_normalization=True)

    # MinMax Scaler, scale values between (0,1)
    normalized = (
        (spectrogram - np.min(spectrogram)) /
        (np.max(spectrogram) - np.min(spectrogram))
    )

    # Rotate 90deg
    normalized = np.swapaxes(normalized, 0, 1)

    # Reshape, tensor 3d
    (height, width) = normalized.shape
    normalized = normalized.reshape(height, width, COLOR_DEPTH)

    assert normalized.dtype == DATA_TYPE
    assert np.max(normalized) == 1.0
    assert np.min(normalized) == 0.0

    return normalized


def main(input_dir, output_dir='./build/folds', split_count=2):
    start = time.time()

    shutil.rmtree(output_dir) 
    language_set = set()
    print("input_dir ", input_dir)
    generate_folds(
        input_dir,
        '.fb.npz',
        output_dir=output_dir,
        group='test',
        input_shape=(WIDTH, FB_HEIGHT),
        normalize=normalize_fb,
        split_count=split_count,
        output_shape=(FB_HEIGHT, WIDTH, COLOR_DEPTH),
        language_set=language_set
    )
    generate_folds(
        input_dir,
        '.fb.npz',
        output_dir=output_dir,
        group='train',
        input_shape=(WIDTH, FB_HEIGHT),
        normalize=normalize_fb,
        split_count=2,
        output_shape=(FB_HEIGHT, WIDTH, COLOR_DEPTH),
        language_set=language_set
    )


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    language_list = sorted(list(language_set))
    language_label_file = os.path.join(output_dir, 'language_label.txt')
    with open(language_label_file, 'w') as f:
        f.write(','.join(language_list))

    end = time.time()
    print("It took [s]: ", end - start)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate various features from audio samples.')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        help="Directory containing data",
        default="./build",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="output data dir",
        default="./build/folds",
    )
    parser.add_argument(
        "-s",
        "--split_count",
        type=int,
        default=2,
        help="训练数据分成几部分"
    )

    parser.set_defaults(debug=False)

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args.input_dir, args.output_dir, args.split_count)




