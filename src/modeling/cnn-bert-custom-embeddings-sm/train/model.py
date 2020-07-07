import os
import pandas as pd
import re
import string
import numpy as np
import pickle
import math
import random
import argparse
import json

import boto3
from sagemaker import get_execution_role

import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers
import tensorflow_hub as hub
from tensorflow.keras import layers


import bert
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
from sklearn.utils import class_weight
import joblib
from sklearn.model_selection import train_test_split

# Dictionary of English Contractions
contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have","she's":"she is","he's":"he is"}

# Regular expression for finding contractions
contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                trainable=True)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)


# Function for expanding contractions
def expand_contractions(text,contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

def preprocess_text(sen):
    #convert ’ to '
    
    sentence = sen.replace("’","'")
    
    #expand contractions 
    sentence = expand_contractions(sentence.lower())
    # Removing punctuation
    #sentence = re.sub('<[^>]+>',' ', sen)

    # Remove punctuations and numbers and foreign characters
    sentence = re.sub('[^a-zA-Z]',' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+",' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+',' ', sentence)

    return sentence

def create_model(max_seq_len,cnn_filters,dropout_rate,dnn_units):
    input_ids = keras.layers.Input(
                                   shape=(max_seq_len, ),
                                   dtype='int32',
                                   name="input_ids"
                                   )
    input_mask = keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name="input_mask")
    segment_ids = keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name="segment_ids")
    
    _, bert_output = bert_layer([input_ids, input_mask, segment_ids])
    #bert_output = bert_layer(input_ids)
    print("bert shape", bert_output.shape)
    
    
    cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")(bert_output)
    cnn_layer1 = layers.GlobalMaxPool1D()(cnn_layer1)
    cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")(bert_output)
    cnn_layer2 = layers.GlobalMaxPool1D()(cnn_layer2)
    cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")(bert_output)
    cnn_layer3 = layers.GlobalMaxPool1D()(cnn_layer3)

    concatenated = tf.concat([cnn_layer1, cnn_layer2, cnn_layer3], axis=-1) # (batch_size, 3 * cnn_filters)   
    dense_1 = layers.Dense(units=dnn_units, activation="relu")(concatenated)
    dropout = layers.Dropout(rate=dropout_rate)(concatenated)
       
    last_dense = layers.Dense(units=1,activation="sigmoid")(dropout)
    
    model = keras.Model(inputs=[input_ids,input_mask,segment_ids], outputs=last_dense) 
    model.build(input_shape=(None, max_seq_len))

    return model

def download_weights(local_weights_file):
    #role = get_execution_role()
    bucket='trainedbertmodelweights'
    data_key = 'BERT_EMBEDDINGS_TRAINABLE_CNN_weights-improvement-19-0.98.hdf5'
    weights_location = 's3://{}/{}'.format(bucket, data_key)
    print(weights_location)

    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, data_key, local_weights_file)

def model(local_pretrained_weights):

    precision_obj = tf.keras.metrics.Precision()
    recall_obj = tf.keras.metrics.Recall()
    #Get model for Experiment3 created
    CNN_FILTERS = 200
    DROPOUT_RATE = 0.1
    DNN_UNITS = 128
    MAX_SEQ_LEN = 60
    model = create_model(MAX_SEQ_LEN,CNN_FILTERS,DROPOUT_RATE,DNN_UNITS)

    model.load_weights(f"./{local_pretrained_weights}")

    model.compile(
                   optimizer=keras.optimizers.Adam(1e-5),
                   loss='binary_crossentropy',
                   metrics=['accuracy',precision_obj,recall_obj]
                 )
    return model

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))
    #parser.add_argument('--pretrained_weights', type=str)

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    
    local_pretrained_weights_file = "pretrained_weights.hdf5"
    download_weights(local_pretrained_weights_file)
    
    classifier_model = model(local_pretrained_weights_file)
    
    # Save tokenizer for inference
    with open('bert_tokenizer.pickle', 'wb') as f:
        pickle.dump(tokenizer, f)
        
    #joblib.dump(classifier_model, os.path.join(args.model_dir, "model.joblib"))
    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        classifier_model.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')
        

    
"""
model_fn
    model_dir: (sting) specifies location of saved model

This function is used by AWS Sagemaker to load the model for deployment. 
It does this by simply loading the model that was saved at the end of the 
__main__ training block above and returning it to be used by the predict_fn
function below.
"""
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

"""
input_fn
    request_body: the body of the request sent to the model. The type can vary.
    request_content_type: (string) specifies the format/variable type of the request

This function is used by AWS Sagemaker to format a request body that is sent to 
the deployed model.
In order to do this, we must transform the request body into a numpy array and
return that array to be used by the predict_fn function below.

Note: Oftentimes, you will have multiple cases in order to
handle various request_content_types. Howver, in this simple case, we are 
only going to accept text/csv and raise an error for all other formats.
"""
def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        claims = [preprocess_text(claim_text)]

        max_seq_len=60
        x_input,masks,segments = [], [],[]
        for new_claim in claims:
            text = tokenizer.tokenize(new_claim)
            text = text[:max_seq_len-2]
            input_sequence = ["[CLS]"] + text + ["[SEP]"]
            pad_len = max_seq_len - len(input_sequence)
            tokens = tokenizer.convert_tokens_to_ids(input_sequence)
            tokens += [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * max_seq_len
            x_input.append(np.array(tokens))
            masks.append(np.array(pad_masks))
            segments.append(np.array(segment_ids))
            
        return [x_input, masks, segments]
            
    else:
        raise ValueError("Thie model only supports text/json input")

"""
predict_fn
    input_data: (numpy array) returned array from input_fn above 
    model (sklearn model) returned model loaded from model_fn above

This function is used by AWS Sagemaker to make the prediction on the data
formatted by the input_fn above using the trained model.
"""
def predict_fn(input_data, model):
    return model.predict(input_data)

"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: (string) the content type the endpoint expects to be returned

This function reformats the predictions returned from predict_fn to the final
format that will be returned as the API call response.

Note: While we don't use content_type in this example, oftentimes you will use
that argument to handle different expected return types.
"""
def output_fn(prediction, content_type):
    return prediction