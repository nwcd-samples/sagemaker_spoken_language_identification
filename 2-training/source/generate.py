import numpy as np
import os

from jobs.transcoder import Transcoder
from jobs.file_remover import FileRemover
from jobs.suffix_remover import SuffixRemover
from jobs.normalizer import Normalizer
from jobs.splitter import Splitter
from jobs.speed_deformer import SpeedDeformer
from jobs.pitch_deformer import PitchDeformer
from jobs.noise_deformer import NoiseDeformer
from jobs.pipeline import Pipeline
import shutil


SUFFIXES = [Transcoder.SUFFIX, Normalizer.SUFFIX]

# 8 speeds between (0.8, 1.2); remove the speed with value 1
SPEEDS = np.delete(np.linspace(0.8, 1.2, 9), 4)

# 8 semitones between (-200, 200); remove the semitone with value 0
SEMITONES = np.delete(np.linspace(-200, 200, 9), 4)

OFFSET_IN_SEC = 30
FRAGMENT_DURATION_IN_SEC = 10
NOISE_DURATION_IN_SEC = 30
TRAIN_DURATION_IN_SEC = 5 * 60
TEST_DURATION_IN_SEC = 15 * 60


# 把语音进行分段， 加噪音， 正则化， 加快或者减慢速度等操作， 生成更多的训练数据。
def main(input_dir):

    pipeline = Pipeline(jobs=[

        # prepare noises


        Transcoder(
            input_dir=os.path.join(input_dir, 'noises'),
            output_files_key='noise_transcoder_files',
            codec='flac'),
        Normalizer(
            input_files_key='noise_transcoder_files',
            output_files_key='noise_normalizer_files',
            duration_in_sec=NOISE_DURATION_IN_SEC),
        Splitter(
            input_files_key='noise_normalizer_files',
            output_files_key='noise_splitter_files',
            duration_in_sec=NOISE_DURATION_IN_SEC,
            fragment_duration_in_sec=FRAGMENT_DURATION_IN_SEC),
        SuffixRemover(
            input_files_key='noise_splitter_files',
            suffixes=SUFFIXES),
        FileRemover(input_files_key='noise_transcoder_files'),
        FileRemover(input_files_key='noise_normalizer_files'),

        # prepare train

        Transcoder(
            input_dir=os.path.join(input_dir, 'train'),
            output_files_key='train_transcoder_files',
            codec='flac'),
        Normalizer(
            input_files_key='train_transcoder_files',
            output_files_key='train_normalizer_files',
            offset_in_sec=OFFSET_IN_SEC,
            duration_in_sec=TRAIN_DURATION_IN_SEC),
        Splitter(
            input_files_key='train_normalizer_files',
            output_files_key='train_splitter_files',
            duration_in_sec=TRAIN_DURATION_IN_SEC,
            fragment_duration_in_sec=FRAGMENT_DURATION_IN_SEC),
        SpeedDeformer(
            input_files_key='train_splitter_files',
            output_files_key='train_speed_deformer_files',
            speeds=SPEEDS,
            fragment_duration_in_sec=FRAGMENT_DURATION_IN_SEC),
        PitchDeformer(
            input_files_key='train_splitter_files',
            output_files_key='train_pitch_deformer_files',
            semitones=SEMITONES),
        NoiseDeformer(
            input_files_key='train_splitter_files',
            output_files_key='train_noise_deformer_files',
            input_noise_files_key='noise_splitter_files'),
        SuffixRemover(input_files_key='train_splitter_files', suffixes=SUFFIXES),
        SuffixRemover(
            input_files_key='train_speed_deformer_files',
            suffixes=SUFFIXES),
        SuffixRemover(
            input_files_key='train_pitch_deformer_files',
            suffixes=SUFFIXES),
        SuffixRemover(
            input_files_key='train_noise_deformer_files',
            suffixes=SUFFIXES),
        FileRemover(input_files_key='train_transcoder_files'),
        FileRemover(input_files_key='train_normalizer_files'),

        # prepare test
        Transcoder(
            input_dir=os.path.join(input_dir, 'test'),
            output_files_key='test_transcoder_files',
            codec='flac'),
        Normalizer(
            input_files_key='test_transcoder_files',
            output_files_key='test_normalizer_files',
            offset_in_sec=OFFSET_IN_SEC,
            duration_in_sec=TEST_DURATION_IN_SEC),
        Splitter(
            input_files_key='test_normalizer_files',
            output_files_key='test_splitter_files',
            duration_in_sec=TEST_DURATION_IN_SEC,
            fragment_duration_in_sec=FRAGMENT_DURATION_IN_SEC),
        SuffixRemover(input_files_key='test_splitter_files', suffixes=SUFFIXES),
        FileRemover(input_files_key='test_transcoder_files'),
        FileRemover(input_files_key='test_normalizer_files'),
        
        
    ])

    pipeline.execute()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate various features from audio samples.')
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        help="Directory containing data",
        default="./raw_data",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="output data dir",
        default="./dataset",
    )
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    parser.set_defaults(debug=False)

    args = parser.parse_args()

    if os.path.exists(args.input_dir):
        shutil.copytree(args.input_dir, args.output_dir)
        main(args.output_dir)

    else:
        print(" 文件路径不正确 {} ".format(args.output_dir))






