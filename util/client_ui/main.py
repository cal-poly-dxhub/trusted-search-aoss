import argparse
import asyncio
import io
import os
import json
import logging
import websockets
import zipfile
import boto3
from botocore.exceptions import ClientError
import requests
from dotenv import load_dotenv
import time
import itertools
import sys

load_dotenv()

logger = logging.getLogger(__name__)

REST_X_API_KEY = os.environ.get("REST_X_API_KEY")
REST_API_ENDPOINT = os.environ.get("REST_API_ENDPOINT")
WEBSOCKET_X_API_KEY = os.environ.get("WEBSOCKET_X_API_KEY")
WEBSOCKET_API_NAME = "core-websocket-api"


# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler('client.log')
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
#logger.addHandler(console_handler)




class ApiGatewayWebsocket:
    """Encapsulates Amazon API Gateway websocket functions."""

    def __init__(self, api_name, apig2_client):
        """
        :param api_name: The name of the websocket API.
        :param apig2_client: A Boto3 API Gateway V2 client.
        """
        self.apig2_client = apig2_client
        self.api_name = api_name
        self.api_id = None
        self.api_endpoint = None
        self.api_arn = None
        self.stage = None

    def get_websocket_api_info(self):
        """
        Gets data about a websocket API by name. This function scans API Gateway
        APIs in the current account and selects the first one that matches the
        API name.

        :return: The ID and endpoint URI of the named API.
        """
        self.api_id = None
        paginator = self.apig2_client.get_paginator("get_apis")
        for page in paginator.paginate():
            for item in page["Items"]:
                if item["Name"] == self.api_name:
                    self.api_id = item["ApiId"]
                    self.api_endpoint = item["ApiEndpoint"]
                    return self.api_id, self.api_endpoint
        raise ValueError

websocket_headers = {
    "x-api-key": WEBSOCKET_X_API_KEY
}
rest_headers = {
    "x-api-key": REST_X_API_KEY
}


# Define an async helper function to create the WebSocket connection
async def client_ui(api_endpoint):
    logger.info(api_endpoint)
    processing_done=False

    async def spinner():
        spinner = itertools.cycle(['-', '/', '|', '\\'])
        print("Please wait ",end="")
        while not processing_done:
            sys.stdout.write(next(spinner))
            sys.stdout.flush()
            await asyncio.sleep(0.1)
            sys.stdout.write('\b')

    async def open_connection():
        # Create a WebSocket connection
        async with websockets.connect(api_endpoint, extra_headers=websocket_headers) as websocket:
            message=''
            try:
                loop_control=True
                answer=""
                print("Answer: ", end='')
                while(loop_control):
                    message = await websocket.recv()
                    try:
                        json_object  = json.loads(message)
                        json_object["Payload"]["body"] = json.loads(json_object["Payload"]["body"])
                        json_formatted_str = json.dumps(json_object, indent=2)
                        logger.info("Streamed Answer: %s", answer)
                        logger.info("++++++++++++++++++++++++++++++++++++++++")
                        logger.info("          RECEIVED PAYLOAD              ")
                        logger.info("++++++++++++++++++++++++++++++++++++++++")
                        logger.info("%s",json_formatted_str)
                        if( answer == "" ):
                            print( json_object["Payload"]["body"]["search_answer"])
                        loop_control=False
                    except: # this is streaming text
                        answer+=message
                        print(message, end='')
            except Exception as e:
                logger.info("++++++++++++++++++++++++++++++++++++++++")
                logger.info(f"An error occurred: {e}")
            finally:
                logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                logger.info("WebSocket connection closed.")

    await asyncio.gather(
        open_connection()
    )

def main():
    # set for your query.
    questions=[
        "Did barbie win an oscar?",
        "Where is home alones house?",
        "What city is home alone home?"
    ]

    if( REST_X_API_KEY is None):
        raise ValueError("REST_SOCKET_X_API_KEY not set")
    if( WEBSOCKET_X_API_KEY is None):
        raise ValueError("WEB_SOCKET_X_API_KEY not set")
    if( REST_API_ENDPOINT is None ):
        raise ValueError("REST_API_ENDPOINT not set")
    
    BUILT_ENDPOINT=REST_API_ENDPOINT+"api/async/search"
    print("Built REST URL: ", BUILT_ENDPOINT)


    for USER_INPUT in questions:
        logger.info("User input: %s", USER_INPUT)

        print("=========================")
        print("User input: ", USER_INPUT)

        # JSON payload
        # possible choices for mode are "STREAM" and "DEFAULT"
        # stream will stream the bedrock response for the answer, default will to a typical non streaming bedrock invocation
        payload = {
            "user_input": USER_INPUT,
            "search_size": 10,
            "bedrock_mode": "STREAM"
        }
        # Send the POST request with JSON payload
        response = requests.post(BUILT_ENDPOINT, json=payload, headers=rest_headers)
        data = response.json()
        execution_arn = data["executionArn"]
        logger.info("Started stated machine: %s", execution_arn)
        sock_gateway = ApiGatewayWebsocket(WEBSOCKET_API_NAME, boto3.client("apigatewayv2"))

        logger.info("Starting websocket Client UI.")
        _, api_endpoint = sock_gateway.get_websocket_api_info()

        asyncio.run(client_ui(f"{api_endpoint}/prod?execution_arn={execution_arn}"))

if __name__ == "__main__":
    main()