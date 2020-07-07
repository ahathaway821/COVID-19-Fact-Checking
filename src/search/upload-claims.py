import sys
from requests_aws4auth import AWS4Auth
import boto3
import requests
import time
import pandas as pd
from pprint import pprint

host = 'http://localhost:9200/' # The domain with https:// and trailing slash. For example, https://my-test-domain.us-east-1.es.amazonaws.com/
#host = 'https://vpc-claim-match-5nmoeqwo3jokdkuptul5mkhhfm.us-west-2.es.amazonaws.com/'
path = 'claim-match/_doc' # the Elasticsearch API endpoint
region = 'us-west-2' # For example, us-west-1

#service = 'es'
#credentials = boto3.Session().get_credentials()
#awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

url = host + path
print(sys.argv[1])
df = pd.read_csv(sys.argv[1])
values = { 'date': time.strftime('%Y-%m-%d', time.gmtime(0))}
df = df.fillna(value=values)
df = df.fillna('')
df['json'] = df.apply(lambda x: x.to_json(), axis=1)

'''
Fields available
- RecordNumber
- claim
- label
- source_label
- source
- date
- claim_source
- explanation
- fact_check_url
- label_binary
'''
for i in df.index:
    claim_record = df.loc[i]
    payload = {
        "claim": claim_record["claim"],
        "date": claim_record["date"],
        "label": claim_record["label"],
        "claim_source": claim_record["claim_source"],
        "fact_check_url": claim_record["fact_check_url"],
        "explanation": claim_record["explanation"],
        "clean_claim": claim_record["clean_claim"],
    }
    r = requests.post(url, json=payload) 
    #r = requests.post(url, auth=awsauth, json=payload)
    if r.status_code != 201:
        print('---Error---')
        pprint(vars(r))
        print('---Record---')
        pprint(payload)
        break

    if i % 100 == 0:
        print(f'{i} records uploaded')

print('--- Completed upload ---')
print(f'{df.index} records uploaded successfully')


#python3 upload-claims.py ~/Downloads/claims.csv
