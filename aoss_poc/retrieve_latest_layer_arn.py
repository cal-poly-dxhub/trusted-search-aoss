from constructs import Construct
from aws_cdk import (
    Stack,
    aws_lambda as _lambda,
    custom_resources as cr,
    aws_iam as iam
)
from datetime import datetime

class RetrieveLatestLayerARN(Stack):
    def __init__(self,scope: Construct, construct_id: str, LAYER_NAME:str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        # Lambda function to handle custom resource operations

        get_parameter = cr.AwsCustomResource(self, "GetLatestLayerARN",
            on_update=cr.AwsSdkCall( # will also be called for a CREATE event
                service="Lambda",
                action="ListLayerVersionsCommand",
                parameters={
                    "LayerName": LAYER_NAME,
                    "ForceRun": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                physical_resource_id=cr.PhysicalResourceId.of("GetLatestLayerARN-CR")
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements(
                statements=[
                    iam.PolicyStatement(
                        actions=[
                            "lambda:ListLayerVersions"
                        ],
                        resources=["*"]
                    )
                ]            
            )
        )
        


        # Export latest layer version
        layer_arn = get_parameter.get_response_field("LayerVersions.0.LayerVersionArn")
        self.layer_arn = layer_arn