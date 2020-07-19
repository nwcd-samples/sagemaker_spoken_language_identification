# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import sys
import signal
import traceback
import boto3
import flask

import time
import datetime
import json
from io import StringIO

import numpy as np

import glob
import shutil
import tempfile
import common
import features
import folds
import keras.models
import tensorflow as tf
from audio_toolbox import ffmpeg, sox
from constants import *
from tensorflow.python.keras.backend import set_session

prefix = '/opt/ml/model'
model_path = os.path.join(prefix, 'model.h5')
s3_client = boto3.client('s3')

# The flask app for serving predictions
app = flask.Flask(__name__)

sess = tf.Session()
graph = tf.get_default_graph() #
set_session(sess)
model = keras.models.load_model(model_path) 
model._make_predict_function()

def normalize(input_file):
    temp_dir = tempfile.mkdtemp()

    transcoded_file = os.path.join(temp_dir, 'transcoded.flac')
    ffmpeg.transcode(input_file, transcoded_file)
    
    args_keep_silence = False
    args_silence_min_duration_sec = 0.1
    args_silence_threshold = 0.5
    
    if not args_keep_silence:
        trimmed_file = os.path.join(temp_dir, 'trimmed.flac')
        sox.remove_silence(
            transcoded_file,
            trimmed_file,
            min_duration_sec=args_silence_min_duration_sec,
            threshold=args_silence_threshold)
    else:
        trimmed_file = transcoded_file

    duration = sox.get_duration(trimmed_file)
    duration = int((duration // FRAGMENT_DURATION) * FRAGMENT_DURATION)

    normalized_file = os.path.join(temp_dir, 'normalized.flac')
    sox.normalize(trimmed_file, normalized_file, duration_in_sec=duration)
    return normalized_file, temp_dir


def load_samples(normalized_file, output_dir):
    temp_dir = output_dir
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    fragmented_file = os.path.join(temp_dir, 'fragment@n.flac')
    sox.split(normalized_file, fragmented_file, FRAGMENT_DURATION)
    features.process_audio(temp_dir, output_dir)

    samples = []

    for file in glob.glob(os.path.join(temp_dir, '*.npz')):
        sample = np.load(file)[DATA_KEY]
        sample = folds.normalize_fb(sample)

        assert sample.shape == INPUT_SHAPE
        assert sample.dtype == DATA_TYPE
        samples.append(sample)

    samples = np.array(samples)

    return samples, temp_dir


def predict(model,languages,samples):
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        results = model.predict(samples)
    
    scores = np.zeros(languages)
    ave_scores = np.zeros(languages)

    for result in results:
        ave_scores = ave_scores + result
        scores[np.argmax(result)] += 1
        
    ave_scores = ave_scores / len(samples)
    return scores, results, ave_scores


def clean(paths):
    for path in paths:
        shutil.rmtree(path)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = boto3.client('s3') is not None  # You can insert a health check here

    status = 200 if health else 404
#     status = 200
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference
    """
    
    data = None
    #解析json，
    if flask.request.content_type == 'application/json':
        data = flask.request.data.decode('utf-8')
        data = json.loads(data)
        
        bucket = data['bucket']
        audio_uri = data['audio_uri']
        class_count = int(data['class_count'])
    else:
        return flask.Response(response='This predictor only supports JSON data', status=415, mimetype='text/plain')    

    download_file_name = audio_uri.split('/')[-1]
    s3_client.download_file(bucket, audio_uri, download_file_name)

    tt = time.mktime(datetime.datetime.now().timetuple())

    args_verbose = False
    args_output_dir = './'+ str(int(tt)) + download_file_name.split('.')[0]
    args_input_file = download_file_name


    if not args_verbose:

        # supress all warnings
        import warnings
        warnings.filterwarnings("ignore")

        # supress tensorflow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if not os.path.exists(args_output_dir):
        os.makedirs(args_output_dir)

    if not os.path.exists(args_input_file):
        print('Error, no such file!')
        pass

    print(" parse file path: {} ".format(args_input_file))

## 
    
    try: 
        normalized_file, normalized_dir = normalize(args_input_file)
        
        file_out_dir = os.path.join(args_output_dir , args_input_file.split('/')[-1].split('.')[0]) 

        samples, samples_dir = load_samples(normalized_file, file_out_dir)
        
        scores, results, ave_scores = predict(model, class_count,samples)

        inference_result = {
            'score': ave_scores.tolist(),
            'language_index':np.argmax(ave_scores, axis=-1).tolist()
        }
        _payload = json.dumps(inference_result)
        print('result is ',_payload)
   
    except Exception as e:
        print(e)

    shutil.rmtree(args_output_dir)  
    return flask.Response(response=_payload, status=200, mimetype='application/json')