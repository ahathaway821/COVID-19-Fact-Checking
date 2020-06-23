from requests_aws4auth import AWS4Auth
import boto3
import requests
import pprint

#host = 'http://localhost:9200/' # The domain with https:// and trailing slash. For example, https://my-test-domain.us-east-1.es.amazonaws.com/
host = 'https://vpc-claim-match-5nmoeqwo3jokdkuptul5mkhhfm.us-west-2.es.amazonaws.com/'
path = 'claim-match' # the Elasticsearch API endpoint
region = 'us-west-2' # For example, us-west-1

service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

url = host + path

r = requests.delete(url, auth=awsauth) # requests.get, post, and delete have similar syntax
#r = requests.delete(url, json=payload)

pprint.pprint(vars(r))
