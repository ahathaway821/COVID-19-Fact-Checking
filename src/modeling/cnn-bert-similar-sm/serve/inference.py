import os
import re
import numpy as np
import pickle
import boto3
import json
import sys
import urllib.parse
import requests

import tensorflow as tf

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

s3 = boto3.resource('s3')
local_pickle_filename = 'bert_tokenizer.pickle'
s3_pickle_filename = 'bert_tokenizer_similar.pickle'
with open(local_pickle_filename, 'wb') as data:
    s3.Bucket("covid-19-claims").download_fileobj(s3_pickle_filename, data)


# open a file, where you stored the pickled data
file = open(f'./{local_pickle_filename}', 'rb')
# dump information to that file
tokenizer = pickle.load(file)
# close the file
file.close()

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


def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/json':
        body_raw = data.read().decode('utf-8')
        print(body_raw)
        body = json.loads(body_raw)
        print(body)
        claims = [preprocess_text(body['claim_text'])]
        print(claims)

        max_seq_len=202
        x_input,masks,segments = [],[],[]
        for new_claim in claims:
            similar_claim1 = new_claim
            similar_claim2 = new_claim
            similar_claim3 = new_claim
            similar_claim4 = new_claim
            similar_claim5 = new_claim
            similar_claim6 = new_claim
            similar_claim7 = new_claim
            similar_claim8 = new_claim
            similar_claim9 = new_claim
            try:
                payload = {'claim': urllib.parse.quote(new_claim)}
                r = requests.get('https://88rrgid4rl.execute-api.us-west-2.amazonaws.com/similar-claims?claim='\
                                 +urllib.parse.quote(new_claim)
                                )
                results = json.loads(r.text) 
                if results.get('claim') is not None:
                    similar_claim1 = results.get("similar_claims")[0]['clean_claim']
                    similar_claim2 = results.get("similar_claims")[1]['clean_claim']
                    similar_claim3 = results.get("similar_claims")[2]['clean_claim']
                    similar_claim4 = results.get("similar_claims")[3]['clean_claim']
                    similar_claim5 = results.get("similar_claims")[4]['clean_claim']
                    similar_claim6 = results.get("similar_claims")[5]['clean_claim']
                    similar_claim7 = results.get("similar_claims")[6]['clean_claim']
                    similar_claim8 = results.get("similar_claims")[7]['clean_claim']
                    similar_claim9 = results.get("similar_claims")[8]['clean_claim']
                else: 
                    #try one more time 
                    payload = {'claim': urllib.parse.quote(new_claim)}
                    r = requests.get('https://88rrgid4rl.execute-api.us-west-2.amazonaws.com/similar-claims?claim='\
                                 +urllib.parse.quote(new_claim)
                                )
                    results = json.loads(r.text) 
                    if results.get('claim') is not None:
                        similar_claim1 = results.get("similar_claims")[0]['clean_claim']
                        similar_claim2 = results.get("similar_claims")[1]['clean_claim']
                        similar_claim3 = results.get("similar_claims")[2]['clean_claim']
                        similar_claim4 = results.get("similar_claims")[3]['clean_claim']
                        similar_claim5 = results.get("similar_claims")[4]['clean_claim']
                        similar_claim6 = results.get("similar_claims")[5]['clean_claim']
                        similar_claim7 = results.get("similar_claims")[6]['clean_claim']
                        similar_claim8 = results.get("similar_claims")[7]['clean_claim']
                        similar_claim9 = results.get("similar_claims")[8]['clean_claim']
            except:

                #try once more 
                try:
                    payload = {'claim': urllib.parse.quote(new_claim)}
                    r = requests.get('https://88rrgid4rl.execute-api.us-west-2.amazonaws.com/similar-claims?claim='\
                                     +urllib.parse.quote(new_claim)
                                    )
                    results = json.loads(r.text) 
                    if results.get('claim') is not None:
                        similar_claim1 = results.get("similar_claims")[0]['clean_claim']
                        similar_claim2 = results.get("similar_claims")[1]['clean_claim']
                        similar_claim3 = results.get("similar_claims")[2]['clean_claim']
                        similar_claim4 = results.get("similar_claims")[3]['clean_claim']
                        similar_claim5 = results.get("similar_claims")[4]['clean_claim']
                        similar_claim6 = results.get("similar_claims")[5]['clean_claim']
                        similar_claim7 = results.get("similar_claims")[6]['clean_claim']
                        similar_claim8 = results.get("similar_claims")[7]['clean_claim']
                        similar_claim9 = results.get("similar_claims")[8]['clean_claim']
                except:

                     e = sys.exc_info()[0]


            text = tokenizer.tokenize(new_claim)

            similar_claim1 = tokenizer.tokenize(similar_claim1)
            similar_claim2 = tokenizer.tokenize(similar_claim2)
            similar_claim3 = tokenizer.tokenize(similar_claim3)
            similar_claim4 = tokenizer.tokenize(similar_claim4)
            similar_claim5 = tokenizer.tokenize(similar_claim5)
            similar_claim6 = tokenizer.tokenize(similar_claim6)
            similar_claim7 = tokenizer.tokenize(similar_claim7)
            similar_claim8 = tokenizer.tokenize(similar_claim8)
            similar_claim9 = tokenizer.tokenize(similar_claim9)

            text = text[:20]
            similar_claim1 = similar_claim1[:20]
            similar_claim2 = similar_claim2[:20]
            similar_claim3 = similar_claim3[:20]
            similar_claim4 = similar_claim4[:20]
            similar_claim5 = similar_claim5[:20]
            similar_claim6 = similar_claim6[:20]
            similar_claim7 = similar_claim7[:20]
            similar_claim8 = similar_claim8[:20]
            similar_claim9 = similar_claim9[:20]
            input_sequence = ["[CLS]"] +\
                             text +\
                             similar_claim1 +\
                             similar_claim2 +\
                             similar_claim3 +\
                             similar_claim4 +\
                             similar_claim5 +\
                             similar_claim6 +\
                             similar_claim7 +\
                             similar_claim8 +\
                             similar_claim9 +\
                             ["[SEP]"]



            pad_len = max_seq_len - len(input_sequence)
            tokens = tokenizer.convert_tokens_to_ids(input_sequence)
            tokens += [0] * pad_len
            pad_masks = [1] * len(input_sequence) + [0] * pad_len
            segment_ids = [0] * max_seq_len
            x_input.append(np.array(tokens))
            masks.append(np.array(pad_masks))
            segments.append(np.array(segment_ids))
            

        return json.dumps({"instances": [{
            "input_ids": tokens,
            "input_mask": pad_masks,
            "segment_ids": segment_ids
        }]})
            
    else:
        raise ValueError("Thie model only supports 'application/json' input")

def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type