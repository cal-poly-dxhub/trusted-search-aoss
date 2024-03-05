import json
import boto3
import os

CORS_HEADERS = {
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Origin': os.environ["CORS_ALLOW_UI"] if os.environ["LOCALHOST_ORIGIN"] == "" else os.environ["LOCALHOST_ORIGIN"],
    'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
}
AOSS_ENDPOINT = os.environ["AOSS_ENDPOINT"]

def handler(event,context):
    try:       
        return {
            "statusCode":200,
            "headers": CORS_HEADERS,
            "body": json.dumps({"Hi world":AOSS_ENDPOINT})
        }
    except Exception as e:
        return {
            "statusCode":500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"msg":str(e)})
        }