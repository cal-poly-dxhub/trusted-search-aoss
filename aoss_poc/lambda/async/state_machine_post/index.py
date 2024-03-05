import json
import boto3
import os
from datetime import datetime
import logging

logger = logging.getLogger()
logger.setLevel("INFO")

CORS_HEADERS = {
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Origin': os.environ["CORS_ALLOW_UI"] if os.environ["LOCALHOST_ORIGIN"] == "" else os.environ["LOCALHOST_ORIGIN"],
    'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
}
STATE_MACHINE_ARN = os.environ["STATE_MACHINE_ARN"]

client = boto3.client('stepfunctions')

TABLE_NAME = os.environ["TABLE_NAME"]
dynamodb_resource = boto3.resource('dynamodb')
connections_table = dynamodb_resource.Table(TABLE_NAME)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()

        return json.JSONEncoder.default(self, o)
    

def handler(event,context):
    print(event)
    print(context)

    try:
        user_input=event.get("body")

        response = client.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            input=user_input
        )

        doc_key = {
            'execution_arn':response["executionArn"],
            'connect_id':''
        }
        logging.info(doc_key)
        connections_table.put_item(Item=doc_key)

        return {
            "statusCode":200,
            "headers": CORS_HEADERS,
            "body": json.dumps(response, cls=DateTimeEncoder)
        }
    except Exception as e:
        return {
            "statusCode":500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"msg":str(e)})
        }