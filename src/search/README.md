# Elasticsearch Setup
For each of the following scripts, be sure to add in aws credentials when loading data into aws vs local elasticsearch cluster

## Index
python3 create-index.py 

## Upload data
python3 upload-claims.py path-to-claims.csv

## Test uploaded data
python3 test-query.py "coronavirus"