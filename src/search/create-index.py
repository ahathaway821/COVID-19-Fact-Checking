from requests_aws4auth import AWS4Auth
import boto3
import requests
from pprint import pprint
import time

host = 'http://localhost:9200/' # The domain with https:// and trailing slash. For example, https://my-test-domain.us-east-1.es.amazonaws.com/
#host = 'https://vpc-claim-match-5nmoeqwo3jokdkuptul5mkhhfm.us-west-2.es.amazonaws.com/'
path = 'claim-match' # the Elasticsearch API endpoint
region = 'us-west-2' # For example, us-west-1

#service = 'es'
#credentials = boto3.Session().get_credentials()
#awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

url = host + path

payload = {
  "settings": {
    "index": {
      "analysis": {
        "analyzer": {
          "autocomplete": {
            "tokenizer": "autocomplete",
            "filter": [
              "lowercase"
            ]
          },
          "autocomplete_search": {
            "tokenizer": "lowercase"
          }
        },
        "tokenizer": {
          "autocomplete": {
            "type": "edge_ngram",
            "min_gram": 3,
            "max_gram": 25,
            "token_chars": [
              "letter"
            ]
          }
        } 
      }
    }
  },
  "mappings": {
      "properties": {
        "claim": {
          "type": "text",
          "analyzer": "autocomplete",
          "search_analyzer": "autocomplete_search"
        },
        "label": {
          "type": "text"
        },
        "date": {
           "type": "date",
           "null_value": time.strftime('%Y-%m-%d', time.gmtime(0))
        },
        "claim_source": {
           "type": "text"
        },
        "fact_check_url": {
           "type": "text"
        },
        "explanation": {
           "type": "text"
        },
        "clean_claim": {
           "type": "text"
        },
        "source": {
           "type": "text"
        },
        "source_label": {
           "type": "text"
        }
      }
    
  }
}

#r = requests.put(url, auth=awsauth, json=payload) # requests.get, post, and delete have similar syntax
r = requests.put(url, json=payload)

pprint(vars(r))
print('--- Index Information ---')
#r = requests.get(url, auth=awsauth)
r = requests.get(url)
pprint(vars(r))

