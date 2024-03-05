import boto3
import os
import boto3
import json
import time

from io import StringIO
from html.parser import HTMLParser

from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection

# TODO -- Parameterize content vector
# TODO -- error handling for failed records on ingest (they would be marked on dynamo, but never reprocessed atm)


TABLE_NAME = os.environ["TABLE_NAME"]
AOSS_ENDPOINT = os.environ["AOSS_ENDPOINT"]
# paramaterize this in the future
AOSS_INDEX="trusted"

EMBEDDING_MODE="TITAN.TXT"
# probably make this a shared python file so we don't duplicate code
EMBEDDING_CONFIGURATION = {
    "TITAN.TXT": {
        "model_id":"amazon.titan-embed-text-v1",
        "content_type":"application/json",
        "accept":"application/json",
        "dimensions":1536,
        "payload":{
            "inputText":"DUMMY"
        }
    },
    "COHERE.TXT": {
        "model_id":"cohere.embed-english-v3",
        "content_type":"application/json",
        "accept":"application/json",
        "dimensions":512,
        "payload":{
            "texts":[],
            "input_type":"search_document",
            "truncate":"NONE"
        }
    }
}
EMBEDDING_SELECTION=EMBEDDING_CONFIGURATION[EMBEDDING_MODE]

s3_resource = boto3.resource('s3')
aoss_client = boto3.client('opensearchserverless')
bedrock_client = boto3.client(service_name="bedrock-runtime")
dynamodb_resource = boto3.resource('dynamodb')

region = boto3.Session().region_name
service = 'aoss'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
host=AOSS_ENDPOINT.replace("https://", "")
aoss_client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=300
)
doc_event_table = dynamodb_resource.Table(TABLE_NAME)

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def build_body(cleaned_content):
    if(EMBEDDING_MODE=="COHERE.TXT"):
        EMBEDDING_SELECTION["payload"]["texts"] = [cleaned_content]
    elif(EMBEDDING_MODE=="TITAN.TXT"):
        EMBEDDING_SELECTION["payload"]["inputText"] = cleaned_content

    return json.dumps(EMBEDDING_SELECTION["payload"])

def process(message):
    # Get the S3 bucket and object key
    bucket = message['s3']['bucket']['name']
    key = message['s3']['object']['key']
    obj = s3_resource.Object(bucket, key)
    data = obj.get()['Body'].read().decode('utf-8')
    json_data = json.loads(data)

    # we will need to consider mid batch failure errror handling.
    # poc does not consider this
    for item in json_data:
        doc_key = {
            'id': item['id']
        }
        dynamodb_response = doc_event_table.get_item(Key=doc_key,ConsistentRead=True)
        # Check article already processed; skip if it is
        if 'Item' in dynamodb_response:
            print("Skipping: ",item['id'])
        else:
            cleaned_content = strip_tags(item['content'])
            body = build_body(cleaned_content)
            bedrock_response = bedrock_client.invoke_model(
                body=body, 
                modelId=EMBEDDING_SELECTION["model_id"], 
                accept=EMBEDDING_SELECTION["accept"], 
                contentType=EMBEDDING_SELECTION["content_type"]
            )
            bedrock_response_body = json.loads(bedrock_response.get("body").read())
            embedding = bedrock_response_body.get("embedding")

            document = {
                'content-id': item['id'],
                'content-title': item['title'],
                'content-vector': embedding
            }

            ingest_document(document, item['id'])
            doc_event_table.put_item(Item=doc_key)

# AOSS does not support specifying DOC ID
# We'll need additional logic here to avoid ingesting and duplicating a record (table store or something)
def ingest_document(document,doc_id=None):
    print("Processing doc_id: ",doc_id)
    response = aoss_client.index(
        index=AOSS_INDEX,
        body = document
    )
    print(response)
    

def index_check():
    response = aoss_client.indices.exists(AOSS_INDEX)
    print('\Checking index:')
    print(response)
    return response

def index_create():
    response = aoss_client.indices.create(
        AOSS_INDEX,
        body={
            "settings": {
                "index.knn": True
            },
            "mappings": {
                "properties": {
                    "content-vector": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_SELECTION["dimensions"],
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 512,
                                "m": 32
                            }
                        }
                    },
                    "content-title": {
                        "type": "text"
                    }
                }
            }
        })
    
    print('\nCreating index:')
    print(response)


def handler(event,context):
    print(event)
    
    try:
        # check aoss configuration (IE initialized or not)
        index_exists = index_check()
        if( index_exists==False ):
            index_create()
            wait_checks=0
            wait_breaker=5
            # Poll and wait for the index to be created (takes some time)
            while not index_check():
                print(f"Index  is not yet created. Waiting...")
                time.sleep(5)  # Sleep for 5 seconds before checking again
                wait_checks+=1
                if wait_checks >= wait_breaker:
                    raise ValueError("AOSS Index Creation error. Waited to long. Breaking loop.")
    except Exception as e:
        print("AOSS index creation error: " + str(e) )
            
    # process batch of events
    for batch in event['Records']:
        messages = json.loads(batch["body"])
        for message in messages["Records"]:
            process(message=message)
    print("===SQS Invocation Complete===")