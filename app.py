#!/usr/bin/env python3

import aws_cdk as cdk

from aoss_poc.aoss_vector_stack import AOSSVectorStack
from aoss_poc.aoss_iam_stack import AOSSIamStack
from aoss_poc.handler_stack import HandlerStack
from aoss_poc.aoss_layer_stack import AOSSLayerStack
from aoss_poc.retrieve_latest_layer_arn import RetrieveLatestLayerARN

app = cdk.App()

ALLOW_LOCALHOST_ORIGIN=True
LAYER_NAME="layer_bedrock_botocore_sdk"

aoss_layer_stack = AOSSLayerStack(app, "cdk-aoss-layer-stack",LAYER_NAME=LAYER_NAME)
aoss_iam_stack = AOSSIamStack(app, "cdk-aoss-iam-stack")
aoss_stack = AOSSVectorStack(app, "cdk-aoss-vector-stack", 
                             AOSS_ROLE=aoss_iam_stack.aoss_role)
# add in custom permissions to AOSS Role Here to avoid Cyclic dependency
handler_stack = HandlerStack(app, "cdk-handler-stack", 
                       AOSS_ROLE=aoss_iam_stack.aoss_role,
                       AOSS_ENDPOINT=aoss_stack.aoss_endpoint,
                       AOSS_SEARCHES_ENDPOINT=aoss_stack.aoss_searches_endpoint,
                       LAYER_NAME=LAYER_NAME,
                       ALLOW_LOCALHOST_ORIGIN=ALLOW_LOCALHOST_ORIGIN,
                       )
handler_stack.add_dependency(aoss_layer_stack)
handler_stack.add_dependency(aoss_stack)

app.synth()
