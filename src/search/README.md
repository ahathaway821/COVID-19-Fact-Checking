# Elasticsearch Setup
The purpose of this folder is to
    1. Create an elasticsearch index
    2. Upload data into the index
    3. Test that the index is working correctly

For each of the following scripts, be sure to add in aws credentials when loading data into aws vs local elasticsearch cluster

## Index
python3 create-index.py 

## Upload data
python3 upload-claims.py path-to-claims.csv

## Test uploaded data
python3 test-query.py "coronavirus"