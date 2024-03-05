import json
import logging
import boto3
import os
logger = logging.getLogger()
logger.setLevel("INFO")

TABLE_NAME = os.environ["TABLE_NAME"]
dynamodb_resource = boto3.resource('dynamodb')
connections_table = dynamodb_resource.Table(TABLE_NAME)


def handler(event, context):
    connectId = event["requestContext"]["connectionId"]
    domainName = event["requestContext"]["domainName"]
    stageName = event["requestContext"]["stage"]
    queryStringParameters = event["queryStringParameters"]

    connectionInfo = {
        'Connection ID': connectId,
        'Domain Name': domainName,
        'Stage Name': stageName,
        'queryStringParameters': queryStringParameters
    }
    logging.info(connectionInfo)

    try:
        doc_key = {
            'execution_arn':queryStringParameters.get('execution_arn'),
            'connect_id':connectId
        }
        logging.info(doc_key)
        connections_table.put_item(Item=doc_key)
    except Exception as e:
        logging.error(f"Failed to send response: {str(e)}")
        return {'statusCode': 500}

    return {"statusCode": 200}       

        