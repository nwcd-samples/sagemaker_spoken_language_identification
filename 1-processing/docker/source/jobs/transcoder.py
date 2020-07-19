from . import common
from audio_toolbox import ffmpeg
# import ffmpeg
import glob
import os


class Transcoder:
    SUFFIX = '.transcoder'

    def __init__(self, input_dir, output_files_key, codec):
        
        types = ('*.mp3', '*.wav')
        input_files = []
        for files in types:
            input_files.extend(glob.glob(os.path.join(input_dir, files)))
        
        self.input_files = input_files
        self.output_files_key = output_files_key
        self.codec = codec

    def execute(self, context):
        output_files = context[self.output_files_key] = []

        for input_file in self.input_files:
            print(input_file)
            output_file = common.change_extension(input_file, self.codec)
            output_file = common.append_suffix_to_filename(
                output_file, Transcoder.SUFFIX)
            output_files.append(output_file)

            ffmpeg.transcode(input_file, output_file)
            os.remove(input_file)
