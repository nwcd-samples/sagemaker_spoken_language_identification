from . import common
import os

class SuffixRemover:
    def __init__(self, input_files_key, suffixes):
        self.input_files_key = input_files_key
        self.suffixes = suffixes

    def execute(self, context):
        input_files = context[self.input_files_key]
        output_files = []

        for input_file in input_files:
            output_file = input_file
            for suffix in self.suffixes:
                output_file = output_file.replace(suffix, '')
                
            if os.path.exists(input_file):
                output_files.append(output_file)
                common.rename_file(input_file, output_file)
            else:
                print("-----------------[warning] file is not exists ", input_file)

        context[self.input_files_key] = output_files
