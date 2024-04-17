import boto3
import json
import os
import logging
import time

logger = logging.getLogger()
logger.setLevel("INFO")

TABLE_NAME = os.environ["TABLE_NAME"]

ENDPOINT_URL = os.environ["ENDPOINT_URL"]
ENDPOINT_URL=ENDPOINT_URL.replace("wss://", "https://") + "/prod"

dynamodb_resource = boto3.resource('dynamodb')
connections_table = dynamodb_resource.Table(TABLE_NAME)


apigateway = boto3.client('apigatewaymanagementapi',endpoint_url=ENDPOINT_URL)


def handler(event,context):
    print("~~~~event~~~~")
    print(event)
    print("~~~~context~~~~")
    print(context)
    # get connext ID
    doc_key = {
            'execution_arn':event["execution_arn"],
    }
    dynamodb_response = connections_table.get_item(Key=doc_key,ConsistentRead=True)
    connect_id = dynamodb_response.get('Item', {}).get('connect_id')
    print("~~~~connect_id~~~~")
    print(connect_id)

    # Dirty "Hack" to fix race condition of workflow getting to this point before client updates
    # the dynamodb conneciton value. This needs fixed and cleaned for production
    while( len(connect_id) < 1 ):
        print("~~~~waiting for 1s~~~~")
        time.sleep(1)
        dynamodb_response = connections_table.get_item(Key=doc_key,ConsistentRead=True)
        connect_id = dynamodb_response.get('Item', {}).get('connect_id')
        print("~~~~connect_id~~~~")
        print(connect_id)    

    response_body = json.dumps(event["search_results"])
    try:
        apigateway.post_to_connection(
            Data=response_body.encode('utf-8'),
            ConnectionId=connect_id
        )
    except Exception as e:
        logging.error(f"Failed to send response: {str(e)}")
        return {'statusCode': 500}

    return {"statusCode": 200}       
