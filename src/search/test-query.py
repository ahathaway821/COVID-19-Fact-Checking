import sys
from requests_aws4auth import AWS4Auth
import boto3
import requests
import pandas as pd
from pprint import pprint

host = 'http://localhost:9200/' # The domain with https:// and trailing slash. For example, https://my-test-domain.us-east-1.es.amazonaws.com/
#host = 'https://vpc-claim-match-5nmoeqwo3jokdkuptul5mkhhfm.us-west-2.es.amazonaws.com/'
path = 'claim-match/_search' # the Elasticsearch API endpoint
region = 'us-west-2' # For example, us-west-1

#service = 'es'
#credentials = boto3.Session().get_credentials()
#awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

url = host + path
query_text = sys.argv[1]

payload = {
  "query": {
    "match": {
      "claim": {
        "query": query_text,
      }
    }
  },
  "sort": ["_score", {"date": "desc"}]
}

r = requests.post(url, json=payload)
#r = requests.post(url, auth=awsauth, json=payload)
pprint(vars(r));
