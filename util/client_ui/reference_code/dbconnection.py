import psycopg2
import boto3
from psycopg2 import OperationalError
import json


def open_connection_to_db(secret_name):
    credentials = get_db_credentials(secret_name)
    #print(credentials)
    
    # Use the retrieved credentials
    db_name = "DBNAME"
    db_user = credentials["username"]
    db_password = credentials["password"]
    db_host = "HOSTENDPOINT"
    db_port = "5432"
    # surround with try catch   
    try:
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
    except OperationalError as e:
        print(f"Error connecting to database: {e}")
        conn = None 

    return conn

def get_db_credentials(secret_name):
    """Retrieve database credentials from AWS Secrets Manager."""
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except Exception as e:
        raise e
    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            return json.loads(secret)
        else:
            raise Exception("Secret is not a string")
