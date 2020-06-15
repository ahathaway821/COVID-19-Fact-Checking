# Elasticsearch Setup
The purpose of this folder is to
    1. Create an elasticsearch index
    2. Upload data into the index
    3. Test that the index is working correctly

Each of the following scripts is set up to work locally and to work within aws.

Be sure to:
1. update the host to http:localhost:9200 when running locally
2. comment out aws specific credential lines
3. trigger requests without aws creds

There are commented out lines for each of these 3 items.

## Create Index
python3 create-index.py 

## Upload data
python3 upload-claims.py path-to-claims.csv
Currently, the script is set up to use the claims located at: https://covid-19-claims.s3-us-west-2.amazonaws.com/claims.csv

## Test uploaded data
python3 test-query.py "coronavirus"

## Recreate the index
If at any point the index is corrupted, or you want to reload data:
1. Delete the index using 'python3 delete_index.py'
2. Recreate the index and upload the data again
