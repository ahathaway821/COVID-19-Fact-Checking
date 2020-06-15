import boto3
import pandas as pd
import argparse
import numpy as np
import os
import json
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

import pickle


if __name__ =='__main__':
    # Create a parser object to collect the environment variables that are in the
    # default AWS Scikit-learn Docker container.
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()

    # Load data from the location specified by args.train (In this case, an S3 bucket).
    df = pd.read_csv(os.path.join(args.train,'claims.csv'), index_col=0, engine="python")

    train, valid = train_test_split(df, test_size=0.75, random_state=0)
    
    # Seperate input variables and labels.
    train_x = train['claim']
    train_y = train['label']

    valid_x = valid['claim']
    valid_y = valid['label']

    vectorizer = CountVectorizer(strip_accents="ascii", 
                                 lowercase=True, 
                                 stop_words="english")
    
    train_x_cv = vectorizer.fit_transform(train_x)
    valid_x_cv = vectorizer.transform(valid_x)

    model = MultinomialNB()
    model.fit(train_x_cv, train_y)

    #Save the model to the location specified by args.model_dir
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    
    # Save vectorizer for inference
    bucket = 'covid-19-claims'
    region = 'us-west-2'
    key='nb-v1/training-artifacts/nb_vectorizer.pkl'
    pickle_byte_obj = pickle.dumps(vectorizer) 
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket, key).put(Body=pickle_byte_obj)
    
    
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
        s3 = boto3.resource('s3')
        filename = 'nb_vectorizer.pkl'
        with open(filename, 'wb') as data:
            s3.Bucket("covid-19-claims").download_fileobj('nb-v1/training-artifacts/nb_vectorizer.pkl', data)

        with open(filename, 'rb') as data:
            vectorizer = pickle.load(data)
            print(request_body)
            test_x = json.loads(request_body)['claims']
            valid_x_cv = vectorizer.transform(test_x)
            return valid_x_cv
            
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
    return '|'.join([t for t in prediction])