#!/usr/bin/env python3
import json

import aws_cdk as cdk

from aws_cdk import (
  Stack,
  aws_iam as iam
)
from constructs import Construct


class AOSSIamStack(Stack):

  def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
    super().__init__(scope, construct_id, **kwargs)

    #################################################################################
    # Custom lambda execution role for AOSS
    #################################################################################
    aoss_role = iam.Role(self, "aoss-role",
      assumed_by=iam.CompositePrincipal(
        iam.ServicePrincipal("lambda.amazonaws.com")
        )
    )

    #################################################################################
    # Custom lambda execution role permissions to avoid circular dependency issue
    # >>>>> Pair this down for higher regions
    #################################################################################
    aoss_role.attach_inline_policy(iam.Policy(self, "lambda-basic-execution-logging",
        statements=[iam.PolicyStatement(
            actions=["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],
            resources=["*"]
        )             
        ]
    ))
    aoss_role.attach_inline_policy(iam.Policy(self, "lambda-basic-explicit-invoke",
        statements=[iam.PolicyStatement(
            actions=["lambda:InvokeFunction"],
            resources=["*"]
        )             
        ]
    ))
    aoss_role.attach_inline_policy(iam.Policy(self, "lambda-basic-aoss",
        statements=[iam.PolicyStatement(
            actions=["aoss:BatchGetCollection","aoss:APIAccessAll","aoss:DashboardsAccessAll"],
            resources=["*"]
        )             
        ]
    ))
    aoss_role.attach_inline_policy(iam.Policy(self, "bedrock-allow-policy",
        statements=[iam.PolicyStatement(
            actions=["bedrock:InvokeModel"],
            resources=["*"]
        )
        ]
    ))
    aoss_role.attach_inline_policy(iam.Policy(self, "sqs-allow-policy",
        statements=[iam.PolicyStatement(
            actions=["sqs:ReceiveMessage","sqs:DeleteMessage","sqs:GetQueueAttributes"],
            resources=[ "*" ]
        )
        ]
    ))

    self.aoss_role = aoss_role