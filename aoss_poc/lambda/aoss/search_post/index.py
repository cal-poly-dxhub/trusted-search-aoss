import json
import boto3
import os

from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection

from string import Template, ascii_letters, digits
import time
import hashlib
import datetime
import random


# TODO -- Parameterize content vector

CORS_HEADERS = {
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Origin': os.environ["CORS_ALLOW_UI"] if os.environ["LOCALHOST_ORIGIN"] == "" else os.environ["LOCALHOST_ORIGIN"],
    'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
}
AOSS_ENDPOINT = os.environ["AOSS_ENDPOINT"]
AOSS_SEARCHES_ENDPOINT = os.environ["AOSS_SEARCHES_ENDPOINT"]
AOSS_MISSED_ENDPOINT = os.environ["AOSS_MISSED_ENDPOINT"]
# paramaterize this in the future
AOSS_INDEX="trusted"
AOSS_SEARCH_INDEX="user_queries"
AOSS_MISSED_INDEX="missed_queries"

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


BEDROCK_MODE="CLAUDE.TXT"
# probably make this a shared python file so we don't duplicate code
BEDROCK_CONFIGURATION = {
    "CLAUDE.TXT": {
        "model_id":"anthropic.claude-v2",
        "content_type":"application/json",
        "accept":"*/*",
        "payload":{
            "prompt":"",
            "max_tokens_to_sample":0,
            "temperature":0,
            "top_p":0,
            "top_k":0,
            "stop_sequences":["Human:"]
        }
    }
}
BEDROCK_SELECTION=BEDROCK_CONFIGURATION[BEDROCK_MODE]


bedrock_client = boto3.client(service_name="bedrock-runtime")

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
host=AOSS_SEARCHES_ENDPOINT.replace("https://", "")
aoss_searches_client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=300
)
host=AOSS_MISSED_ENDPOINT.replace("https://", "")
aoss_missed_client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=300
)

def build_body(text):
    if(EMBEDDING_MODE=="COHERE.TXT"):
        EMBEDDING_SELECTION["payload"]["texts"] = [text]
    elif(EMBEDDING_MODE=="TITAN.TXT"):
        EMBEDDING_SELECTION["payload"]["inputText"] = text

    return json.dumps(EMBEDDING_SELECTION["payload"])

def generate_embedding(text):
    body = build_body(text)
    bedrock_response = bedrock_client.invoke_model(
        body=body, 
        modelId=EMBEDDING_SELECTION["model_id"], 
        accept=EMBEDDING_SELECTION["accept"], 
        contentType=EMBEDDING_SELECTION["content_type"]
    )
    bedrock_response_body = json.loads(bedrock_response.get("body").read())
    embedding = bedrock_response_body.get("embedding")
    return embedding

def strip_knn_vector(data,strip_field='content-vector'):
    try:
        rebuild = []
        for entry in data['hits']['hits']:
            entry["_source"][strip_field]=[-1]
            rebuild.append(entry)
        data['hits']['hits'] = rebuild
        return data
    except:
        return data
    
def search_aoss(embeddings,search_size):
    query = {
        'size': search_size,
        'query': {
            'knn': {
                "content-vector": {
                    "vector": embeddings,
                    "k": 5
                }
            }
        }
    }    
    
    response = aoss_client.search(
        body = query,
        index = AOSS_INDEX
    )
    return(response)

def hyde(user_input):
    TEMPERATURE=.1
    TOP_P=.8
    MAX_TOKENS_TO_SAMPLE=512
    TOP_K=250

    prompt_string = """Human: Generate a ficticious article that answers the following user prompt: $user_input
    Assistant:
    """
    template = Template(prompt_string)
    prompt = template.substitute(user_input=user_input)

    print(prompt)

    BEDROCK_SELECTION["payload"]["prompt"] = prompt
    BEDROCK_SELECTION["payload"]["temperature"] = TEMPERATURE
    BEDROCK_SELECTION["payload"]["top_k"] = TOP_K
    BEDROCK_SELECTION["payload"]["top_p"] = TOP_P
    BEDROCK_SELECTION["payload"]["max_tokens_to_sample"] = MAX_TOKENS_TO_SAMPLE
    body = json.dumps(BEDROCK_SELECTION["payload"])

    response = bedrock_client.invoke_model(
        body=body, 
        modelId=BEDROCK_SELECTION["model_id"], 
        accept=BEDROCK_SELECTION["accept"], 
        contentType=BEDROCK_SELECTION["content_type"]
    )
    response_body = json.loads(response.get('body').read())

    hyde_generated_text = response_body.get("completion")
    print(hyde_generated_text)

    return hyde_generated_text


def raw(user_input):
    return user_input

# set approach
MODE_LIST = {
    "RAW":raw,
    "HYDE":hyde
    }
MODE="RAW"

def generate_random_string(length):
    """
    Generate a random string of the specified length.
    
    Args:
        length (int): The desired length of the random string.
        
    Returns:
        str: A random string of the specified length.
    """
    # Define the characters to choose from
    characters = ascii_letters + digits
    
    # Generate a random string by joining random characters
    random_string = ''.join(random.choice(characters) for _ in range(length))
    
    return random_string


def insert_missed(question):
    try:
        cis_exists=check_index_missed()
        if( cis_exists==False):
            create_index_missed()
            wait_checks=0
            wait_breaker=5
            # Poll and wait for the index to be created (takes some time)
            while not check_index_missed():
                print(f"Missed index is not yet created. Waiting...")
                time.sleep(5)  # Sleep for 5 seconds before checking again
                wait_checks+=1
                if wait_checks >= wait_breaker:
                    raise ValueError("AOSS Index Creation error. Waited to long. Breaking loop.")
    except Exception as e:
        print("AOSS index creation error: " + str(e) )

    RETRY_CAP=4
    tries=0
    while( tries < RETRY_CAP ):
        try:
            # Generate a timestamp for the current time
            now = datetime.datetime.now(datetime.timezone.utc)
            timestamp = now.isoformat()

            uid = timestamp + question + generate_random_string(10)
            # Create a SHA-256 hash object
            sha256_hash = hashlib.sha256()
            sha256_hash.update(uid.encode('utf-8'))
            digest = sha256_hash.hexdigest()

            document = {
                "timestamp":timestamp,
                "unique_id":digest,
                "query":question,
            }
            print("Ingesting missed query: ",question)
            response = aoss_missed_client.index(
                index=AOSS_MISSED_INDEX,
                body = document
            )
            break
        except Exception as e:
            print(str(e))
            print("Sleeping, missed index has not reached consistency")
            time.sleep(15)
            tries+=1
    if( tries == 3 ):
        raise "Failed to insert missed"


def best_answer(question, search_results):
    TEMPERATURE=0
    TOP_P=.9
    MAX_TOKENS_TO_SAMPLE=2048
    TOP_K=250
    merged_content = ''
    for hit in search_results['hits']['hits']:
        source = hit['_source']
        if 'content-raw-cleaned' in source:
            merged_content += source['content-raw-cleaned'] + '\n'
    data = merged_content

    prompt_string = """Human: You are to answer the question using the data in the following article.  Do not make up your answer, only use 
    supporting data from the article.
    
    If you don't have enough data respond with exactly the following 'I don't have enough information to answer that question.'
    
    Given the following news article data [ $data ] can you please give a concise answer to the following question. $question
    
    Assistant:
    """
    template = Template(prompt_string)
    prompt = template.substitute(data=data,question=question)

    # print(prompt)

    BEDROCK_SELECTION["payload"]["prompt"] = prompt
    BEDROCK_SELECTION["payload"]["temperature"] = TEMPERATURE
    BEDROCK_SELECTION["payload"]["top_k"] = TOP_K
    BEDROCK_SELECTION["payload"]["top_p"] = TOP_P
    BEDROCK_SELECTION["payload"]["max_tokens_to_sample"] = MAX_TOKENS_TO_SAMPLE
    body = json.dumps(BEDROCK_SELECTION["payload"])

    response = bedrock_client.invoke_model(
        body=body, 
        modelId=BEDROCK_SELECTION["model_id"], 
        accept=BEDROCK_SELECTION["accept"], 
        contentType=BEDROCK_SELECTION["content_type"]
    )
    response_body = json.loads(response.get('body').read())

    answer_text = response_body.get("completion")
    print(answer_text)
    if "I don't have enough information to answer that question." in answer_text:
        #mismatch
        print("++++++++++++RUNNING INSERT INTO MISSED ROUTINE++++++++++++")
        insert_missed(question)
        return (-1,answer_text)
    else:
        return (1,answer_text)

def check_index_searches():
    response = aoss_searches_client.indices.exists(AOSS_SEARCH_INDEX)
    print('\Checking index:')
    print(response)
    return response

def check_index_missed():
    response = aoss_searches_client.indices.exists(AOSS_MISSED_INDEX)
    print('\Checking index:')
    print(response)
    return response

def create_index_missed():
    response = aoss_searches_client.indices.create(
        AOSS_MISSED_INDEX,
        body={
            "settings": {
                "index.knn": True
            },
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "unique_id": {"type": "keyword"},
                    "query": {"type": "text"}
                }
            }
        })
    
    print('\nCreating index:')
    print(response)





def create_index_search():
    response = aoss_searches_client.indices.create(
        AOSS_SEARCH_INDEX,
        body={
            "settings": {
                "index.knn": True
            },
            "mappings": {
                "properties": {
                    "user-query-vector": {
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
                    "user-query": {
                        "type": "text"
                    },
                    "similar-queries": {
                        "type": "object"
                    },
                    "search-results":{
                        "type": "object"
                    },
                    "search-answer":{
                        "type": "text"
                    }
                }
            }
        })
    
    print('\nCreating index:')
    print(response)

def find_nearest_query(user_input,embeddings):
    # can probably add in a correct routine here when you get two hits and remap these docs
    # into a single doc
    query = {
        'size': 1,
        'query': {
            'knn': {
                "user-query-vector": {
                    "vector": embeddings,
                    "k": 5
                }
            }
        }
    }    
    
    response = aoss_searches_client.search(
        body = query,
        index = AOSS_SEARCH_INDEX
    )

    max_score = -1 if response['hits']['max_score'] is None else response['hits']['max_score']
    print("Max_Score: ", max_score)

    print("~~~QUERY SEARCH RESULT~~~")
    print(strip_knn_vector(response,strip_field="user-query-vector"))
    return {
        "hits":len(response['hits']['hits']),
        "max_score":max_score,
        "response":response 
    }
           
def insert_query_result( user_input, embedding, search_results, answer):
    document = {
        "user-query":user_input,
        "user-query-vector":embedding,
        "similar-queries":{},
        "search-results":search_results,
        "search-answer":answer
    }
    print("Ingesting query: ",user_input)
    response = aoss_searches_client.index(
        index=AOSS_SEARCH_INDEX,
        body = document
    )
    print(response)

def insert_similar_query( nearest_query_result, user_input ):
    doc_id=nearest_query_result["response"]["hits"]["hits"][0]["_id"]
    similar_queries=nearest_query_result["response"]["hits"]["hits"][0]["_source"]["similar-queries"]

    try:
        # Create a SHA-256 hash object
        sha256_hash = hashlib.sha256()
        sha256_hash.update(user_input.encode('utf-8'))
        digest = sha256_hash.hexdigest()

        print(f"SHA-256 hash of '{user_input}' is: {digest}")
        similar_queries[digest] = user_input
    except Exception as e:
        print(str(e))

    update_data = {
        "doc": {
            "similar-queries":similar_queries
        }
    }

    print(doc_id)
    print(similar_queries)
    print(update_data)

    response = aoss_searches_client.update(
            index = AOSS_SEARCH_INDEX,
            body = update_data,
            id=doc_id
    )
    print(response)

def handler(event,context):
    print(event)
    print(context)

    # Patch to support parallel testing of synchronous api gateway call and
    # new async route with step functions
    try:
        payload=event.get("body")
        if payload is not None: # come in via API Gateway
            field_values=json.loads(payload)
        else: # come in via step functions
            field_values=event

        user_input = field_values["user_input"]
        search_size = field_values["search_size"]

        user_input = MODE_LIST[MODE](user_input)
        embedding=generate_embedding(user_input)
        # cache logic
        try:
            cis_exists=check_index_searches()
            if( cis_exists==False):
                create_index_search()
                wait_checks=0
                wait_breaker=5
                # Poll and wait for the index to be created (takes some time)
                while not check_index_searches():
                    print(f"Index  is not yet created. Waiting...")
                    time.sleep(5)  # Sleep for 5 seconds before checking again
                    wait_checks+=1
                    if wait_checks >= wait_breaker:
                        raise ValueError("AOSS Index Creation error. Waited to long. Breaking loop.")
        except Exception as e:
            print("AOSS index creation error: " + str(e) )

        RETRY_CAP=4
        tries=0
        while( tries < RETRY_CAP ):
            try:
                nearest_query_result=find_nearest_query(user_input=user_input,embeddings=embedding)
                break
            except Exception as e:
                print(str(e))
                print("Sleeping, index has not reached consistency")
                time.sleep(15)
                tries+=1
        if( tries == 3 ):
            raise "Failed to query"

        SIMILARITY_THRESHOLD=.85 #threshold to consider a query as something that was asked before
        if( nearest_query_result["hits"] == 0 or nearest_query_result["max_score"] < SIMILARITY_THRESHOLD):
            #new user search term, add to queries doc store
            search_results = strip_knn_vector(search_aoss(embeddings=embedding,search_size=search_size))
            print("~~~DOC SEARCH RESULT~~~")
            print(strip_knn_vector(search_results))
            answer_result = best_answer(question=user_input,search_results=search_results)
            found_match=answer_result[0]
            answer=answer_result[1]
            if( found_match == 1 ): # only insert for bypass if we get hits
                insert_query_result( user_input=user_input, embedding=embedding, search_results=search_results, answer=answer)
        else: #hit on similar query; will always be a result of one (top) if we hit threshold
            # map
            print("!!!!!!!!!!!!!!!!!!! BYPASSED BEDROCK CALL !!!!!!!!!!!!!!!!!!!")
            search_results = nearest_query_result["response"]["hits"]["hits"][0]["_source"]["search-results"]
            answer= nearest_query_result["response"]["hits"]["hits"][0]["_source"]["search-answer"]
            # add to similar queries
            if( nearest_query_result["max_score"]==1):
                 print("!!!!!!!!!!!!!!!!!!! BYPASSED SIMILAR STATEMENT INSERT, EXACT MATCH !!!!!!!!!!!!!!!!!!!")
            else:
                insert_similar_query( nearest_query_result=nearest_query_result, user_input=user_input )


        return {
            "statusCode":200,
            "headers": CORS_HEADERS,
            "body": json.dumps({"search_response":search_results,"search_answer":answer})
        }
    except Exception as e:
        return {
            "statusCode":500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"msg":str(e)})
        }