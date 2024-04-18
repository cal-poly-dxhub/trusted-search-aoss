from dotenv import load_dotenv
import os


load_dotenv()

REST_X_API_KEY = os.environ.get("REST_X_API_KEY")
REST_API_ENDPOINT = os.environ.get("REST_API_ENDPOINT")
WEBSOCKET_X_API_KEY = os.environ.get("WEBSOCKET_X_API_KEY")
WEBSOCKET_API_NAME = "core-websocket-api"

print("REST_X_API_KEY: ", REST_X_API_KEY)
print("REST_API_ENDPOINT: ", REST_API_ENDPOINT)
print("WEBSOCKET_X_API_KEY: ", WEBSOCKET_X_API_KEY)
print("WEBSOCKET_API_NAME: ", WEBSOCKET_API_NAME)