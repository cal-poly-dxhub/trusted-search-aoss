import os

from constructs import Construct
from aws_cdk import(
    Stack,
    aws_lambda as lambda_
)

class AOSSLayerStack(Stack):
    def __init__(self,scope: Construct, construct_id: str, LAYER_NAME: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Custom layer to support bedrock calls and aoss calls
        layer_bedrock_boto3_sdk = lambda_.LayerVersion(
            self, "layer_bedrock_botocore_sdk",
            code=lambda_.Code.from_asset(os.path.join("aoss_poc/lambda/custom_packages/layers","aoss_poc.zip")),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_11],
            description="SDK to support bedrock and aoss calls.",
            layer_version_name=LAYER_NAME
        )