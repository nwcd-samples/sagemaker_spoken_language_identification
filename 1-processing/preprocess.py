import time
import re
import sys
import os

sys.path.append("./")

os.system("python3 /opt/ml/code/generate.py --input_dir /opt/ml/processing/input_data --output_dir /opt/dataset")
os.system("python3 /opt/ml/code/features.py --input_dir /opt/dataset --output_dir  /opt/build")
os.system("python3 /opt/ml/code/folds.py --input_dir /opt/build --output_dir /opt/ml/processing/output_data")