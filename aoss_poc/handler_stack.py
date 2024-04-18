import os
import boto3
from constructs import Construct
from aws_cdk import(
    Duration,
    Stack,
    aws_secretsmanager as secretsmanager,
    SecretValue as SecretValue,
    aws_apigateway as apigateway,
    aws_apigatewayv2 as apigatewayv2,
    aws_apigatewayv2_integrations as apigatewayv2_integrations,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_sqs as sqs,
    aws_s3_notifications as s3n,
    custom_resources as cr,
    aws_dynamodb as dynamodb,
    aws_stepfunctions as stepfunctions,
    aws_stepfunctions_tasks as tasks,
)

from datetime import datetime


class HandlerStack(Stack):
    def __init__( self,scope: Construct, 
                 construct_id: str,  
                 AOSS_ROLE:iam.Role, 
                 AOSS_ENDPOINT:str, 
                 AOSS_SEARCHES_ENDPOINT:str,
                 ALLOW_LOCALHOST_ORIGIN:bool,
                 LAYER_NAME:str,
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        LOCALHOST_ORIGIN="*"
        ALLOW_LOCALHOST_ORIGIN=ALLOW_LOCALHOST_ORIGIN


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
        layer_arn = get_parameter.get_response_field("LayerVersions.0.LayerVersionArn")
        # Fix to get rid of circular dependency if layer updates.
        LAMBDA_CUSTOM_LAYER = lambda_.LayerVersion.from_layer_version_arn(self,id="layer_aoss",layer_version_arn=layer_arn)

        #################################################################################
        # Misc setup
        #################################################################################   
        # API Gateway Secret
        apig_api_key_secret = secretsmanager.Secret(
            self, 'apig-secret',
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"apiKey": "supersecretkey"}',
                generate_string_key='apiKey',
                password_length=32,
                exclude_characters='"@/\\',
            )
        )
        # API Gateway Secret for Websockets
        apig_websocket_api_key_secret = secretsmanager.Secret(
            self, 'apig-websocket-secret',
            generate_secret_string=secretsmanager.SecretStringGenerator(
                secret_string_template='{"apiKey": "supersecretkey"}',
                generate_string_key='apiKey',
                password_length=32,
                exclude_characters='"@/\\',
            )
        )
       


        # Main API Gateway
        core_api = apigateway.RestApi(
            self,"core-api",
            endpoint_configuration=apigateway.EndpointConfiguration(
                types=[apigateway.EndpointType.REGIONAL]
            ),
            default_cors_preflight_options=apigateway.CorsOptions(
                allow_methods=['GET', 'OPTIONS','PUT','PATCH','POST'],
                allow_origins=[LOCALHOST_ORIGIN if ALLOW_LOCALHOST_ORIGIN else ""])
        )

        #################################################################################
        # DynamoDB To Prevent Event Dupes of AOSS
        #################################################################################   
        dynamo_articles = dynamodb.Table(self,"dynamo-articles",
            partition_key=dynamodb.Attribute(name="id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST
        )
        dynamo_connections = dynamodb.Table(self,"dynamo-connections",
            partition_key=dynamodb.Attribute(name="execution_arn", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST
        )

        #################################################################################
        # Tester Lambda
        #################################################################################        
        fn_hello_get = lambda_.Function(
            self,"fn-hello-get",
            description="hello-get", #microservice tag
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="index.handler",
            role=AOSS_ROLE,
            code=lambda_.Code.from_asset(os.path.join("aoss_poc/lambda/testing","hello_get")),
            environment={
                "AOSS_ENDPOINT": AOSS_ENDPOINT.value,
                "AOSS_SEARCHES_ENDPOINT": AOSS_SEARCHES_ENDPOINT.value,
                "LOCALHOST_ORIGIN":LOCALHOST_ORIGIN if ALLOW_LOCALHOST_ORIGIN else ""
            },
            layers=[LAMBDA_CUSTOM_LAYER ]
        )

        #################################################################################
        # AOSS Search Functionality
        #################################################################################
        fn_aoss_search_post = lambda_.Function(
            self,"fn-aoss-search-post",
            description="aoss-search-post", #microservice tag
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="index.handler",
            role=AOSS_ROLE,
            code=lambda_.Code.from_asset(os.path.join("aoss_poc/lambda/aoss","search_post")),
            environment={
                "AOSS_ENDPOINT": AOSS_ENDPOINT.value,
                "AOSS_SEARCHES_ENDPOINT": AOSS_SEARCHES_ENDPOINT.value,
                "LOCALHOST_ORIGIN":LOCALHOST_ORIGIN if ALLOW_LOCALHOST_ORIGIN else "",
            },
            timeout=Duration.minutes(5),
            layers=[ LAMBDA_CUSTOM_LAYER ]
        )

        #################################################################################
        # AOSS Ingest Lambda
        #################################################################################        
        fn_ingest_handler = lambda_.Function(
            self,"fn-ingest-handler",
            description="ingest-handler", #microservice tag
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="index.handler",
            role=AOSS_ROLE,
            code=lambda_.Code.from_asset(os.path.join("aoss_poc/lambda/aoss","ingest_handler")),
            environment={
                "AOSS_ENDPOINT": AOSS_ENDPOINT.value,
                "AOSS_SEARCHES_ENDPOINT": AOSS_SEARCHES_ENDPOINT.value,
                "TABLE_NAME": dynamo_articles.table_name,
            },
            timeout=Duration.minutes(5),
            layers=[ LAMBDA_CUSTOM_LAYER ]
        )
        # Create an S3 bucket
        article_bucket = s3.Bucket(self, "article-json")

        # Create an SQS queue
        article_invocation_dlq = sqs.Queue(self, "article-invocation-dlq")
        article_invocation_queue = sqs.Queue(self, "article-invocation-queue",
                                             dead_letter_queue=sqs.DeadLetterQueue(
                                                 max_receive_count=5,
                                                 queue=article_invocation_dlq),
                                             visibility_timeout=Duration.seconds(300))
        article_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED, 
            s3n.SqsDestination(article_invocation_queue), 
            s3.NotificationKeyFilter(suffix='.json')
        )




        #################################################################################
        # ASync Open Connection Handler
        #################################################################################        
        fn_open_connection_handler = lambda_.Function(
            self,"fn-open-connection-handler",
            description="open-connection-handler", #microservice tag
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="index.handler",
            role=AOSS_ROLE,
            code=lambda_.Code.from_asset(os.path.join("aoss_poc/lambda/async","open_connection_handler")),
            environment={
                "TABLE_NAME": dynamo_connections.table_name,
            },
            timeout=Duration.minutes(5),
            layers=[ LAMBDA_CUSTOM_LAYER ]
        )

        connect_integration=apigatewayv2_integrations.WebSocketLambdaIntegration("ConnectIntegration", fn_open_connection_handler)
        websocket_api = apigatewayv2.WebSocketApi(self, "core-websocket-api",
            api_key_selection_expression=apigatewayv2.WebSocketApiKeySelectionExpression.HEADER_X_API_KEY,
            connect_route_options=apigatewayv2.WebSocketRouteOptions(
                integration=connect_integration
            )
        )



        websocket_api_stage = apigatewayv2.WebSocketStage(self, "core-websocket-api-stage",
            web_socket_api=websocket_api,
            stage_name="prod",
            auto_deploy=True
        )
        #websocket_api.add_route("OpenConnection",
        #    integration=apigatewayv2_integrations.WebSocketLambdaIntegration("open-connection-handler-integration",fn_open_connection_handler)
        #)

        #################################################################################
        # Async Issue Callback
        #################################################################################        
        fn_callback_handler = lambda_.Function(
            self,"fn-callback-handler",
            description="callback-handler", #microservice tag
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="index.handler",
            role=AOSS_ROLE,
            code=lambda_.Code.from_asset(os.path.join("aoss_poc/lambda/async","issue_callback")),
            environment={
                "TABLE_NAME": dynamo_connections.table_name,
                "ENDPOINT_URL": websocket_api.api_endpoint
            },
            timeout=Duration.minutes(5),
            layers=[ LAMBDA_CUSTOM_LAYER ]
        )

        #################################################################################
        # State Machine
        #################################################################################             
        invoke_search_handler = tasks.LambdaInvoke(self, "InvokeSearchHandler",
            lambda_function=fn_aoss_search_post,
            payload=stepfunctions.TaskInput.from_object({
                "user_input": stepfunctions.JsonPath.string_at("$.user_input"),
                "search_size": stepfunctions.JsonPath.string_at("$.search_size"),
            }),
            result_path="$.search_results"
        )
        invoke_callback_handler= tasks.LambdaInvoke(self, "InvokeCallbackHandler",
            lambda_function=fn_callback_handler,
            payload=stepfunctions.TaskInput.from_object({
                "execution_arn":  stepfunctions.JsonPath.execution_id,
                "search_results": stepfunctions.JsonPath.string_at("$.search_results")
            }),
        )


        state_start = stepfunctions.Pass(self, "RunSearch")
        state_midpoint = stepfunctions.Pass(self, 
                "RunIssueCallback"
        )
        definition = state_start.next(invoke_search_handler).next(state_midpoint).next(invoke_callback_handler)
        

        search_state_machine = stepfunctions.StateMachine(self, "SearchStateMachine",
            definition_body=stepfunctions.DefinitionBody.from_chainable(definition)
        )

        #################################################################################
        # Sync Call Initiate State Machine
        #################################################################################        
        fn_state_machine_handler = lambda_.Function(
            self,"fn-state-machine-handler",
            description="state-machine-handler", #microservice tag
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="index.handler",
            role=AOSS_ROLE,
            code=lambda_.Code.from_asset(os.path.join("aoss_poc/lambda/async","state_machine_post")),
            environment={
                "STATE_MACHINE_ARN": search_state_machine.state_machine_arn,
                "TABLE_NAME": dynamo_connections.table_name,
                "LOCALHOST_ORIGIN":LOCALHOST_ORIGIN if ALLOW_LOCALHOST_ORIGIN else "",
            },
            timeout=Duration.minutes(5),
            layers=[ LAMBDA_CUSTOM_LAYER ]
        )


        #################################################################################
        # These following lines will not work as they cause circular dependency
        #article_invocation_queue.grant_consume_messages(fn_ingest_handler)
        #article_bucket.grant_read(fn_ingest_handler)
        #fn_ingest_handler.add_event_source(event_sources.SqsEventSource(article_invocation_queue))

        AOSS_ROLE.attach_inline_policy(iam.Policy(self, "s3-allow-policy",
            statements=[iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[ "arn:aws:s3:::" + article_bucket.bucket_name + "/*"]
            )
            ]
        ))
        AOSS_ROLE.attach_inline_policy(iam.Policy(self, "dynamodb-allow-policy",
            statements=[iam.PolicyStatement(
                actions=[
                    "dynamodb:BatchGetItem",
                    "dynamodb:GetRecords",
                    "dynamodb:GetShardIterator",
                    "dynamodb:Query",
                    "dynamodb:GetItem",
                    "dynamodb:Scan",
                    "dynamodb:BatchWriteItem",
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:DeleteItem",
                    "dynamodb:DescribeTable",
                ],
                resources=[ "*"]
            )
            ]
        ))
        AOSS_ROLE.attach_inline_policy(iam.Policy(self, "stepfunction-allow-policy",
            statements=[iam.PolicyStatement(
                actions=[
                     "states:StartExecution" 
                ],
                resources=[ "*"]
            )
            ]
        ))        
        AOSS_ROLE.attach_inline_policy(iam.Policy(self, "manageconnections-allow-policy",
            statements=[iam.PolicyStatement(
                actions=[
                     "execute-api:ManageConnections", 
                      "execute-api:Invoke"
                ],
                resources=[ "*"]
            )
            ]
        ))        


        # Create an event source mapping using CfnEventSourceMapping
        event_source_mapping = lambda_.CfnEventSourceMapping(
            self, "SQSTrigger",
            event_source_arn=article_invocation_queue.queue_arn,
            function_name=fn_ingest_handler.function_name,
            batch_size=5  # Set the batch size as needed
        )


        #################################################################################
        # APIG Routing
        #################################################################################
        ###### Route Base = /api [match for cloud front purposes]
        api_route = core_api.root.add_resource("api")

        ###### Route Base = /api/hello
        pr_hello=api_route.add_resource("hello")
        # GET /hello
        intg_hello_get=apigateway.LambdaIntegration(fn_hello_get)
        method_hello=pr_hello.add_method(
            "GET",intg_hello_get,
            api_key_required=True
        )

        ###### Route Base = /api/aoss
        pr_aoss=api_route.add_resource("aoss")
        ###### Route Base = /api/aoss/search
        pr_aoss_search=pr_aoss.add_resource("search")
        intg_search_post=apigateway.LambdaIntegration(fn_aoss_search_post)
        method_search_post=pr_aoss_search.add_method(
            "POST",intg_search_post,
            api_key_required=True
        )

        ###### Route Base = /api/async
        pr_async=api_route.add_resource("async")
        ###### Route Base = /api/aysnc/search
        pr_async_search=pr_async.add_resource("search")
        intg_async_search_post=apigateway.LambdaIntegration(fn_state_machine_handler)
        method_search_post=pr_async_search.add_method(
            "POST",intg_async_search_post,
            api_key_required=True
        )


        #################################################################################
        # Usage plan and api key to "lock" API
        #################################################################################

        # Grab secrets generated; this line needs visited for non developpment environments
        core_key=core_api.add_api_key("core-apig-key",value=apig_api_key_secret.secret_value_from_json("apiKey").unsafe_unwrap())
                                              

        plan = core_api.add_usage_plan(
            "UsagePlan",name="public plan",
            throttle=apigateway.ThrottleSettings(
                rate_limit=10,
                burst_limit=2
            )
        )
        
        plan.add_api_key(core_key)
        plan.add_api_stage(api=core_api,stage=core_api.deployment_stage)


        #################################################################################
        # Usage plan and api key to "lock" API for websocket
        #################################################################################
        ### Not possible on L2 Constructs atm, we do this via L1
        cfn_usage_plan = apigateway.CfnUsagePlan(self, "WebSocketUsagePlan",
           api_stages=[apigateway.CfnUsagePlan.ApiStageProperty(
               api_id=websocket_api.api_id,
               stage=websocket_api_stage.stage_name,
               #NOTE: Can not be defined as "No method level throttling for Web Socket APIs" can be checked from the UsagePlan console.
               # throttle={
               #     "$connect": apigw.CfnUsagePlan.ThrottleSettingsProperty(
               #         burst_limit=123,
               #         rate_limit=123
               #     )
               # }
           )],
           description="description",
           quota=apigateway.CfnUsagePlan.QuotaSettingsProperty(
               limit=123,
               offset=0,
               period="DAY"
           ),
           throttle=apigateway.CfnUsagePlan.ThrottleSettingsProperty(
               burst_limit=123,
               rate_limit=123
           ),
           usage_plan_name="WebsocketUsagePlan"
        )

        cfn_api_key = apigateway.CfnApiKey(self, "WebSocketApiKey",
               description="Key for websocket connections",
               enabled=True,
               name="WebsocketApiKey",
               # stage_keys=[apigw.CfnApiKey.StageKeyProperty(
               #     rest_api_id= api.api_id,
               #     stage_name= stage.stage_name
               # )],
               value=apig_websocket_api_key_secret.secret_value_from_json("apiKey").unsafe_unwrap()
        )
    
        cfn_usage_plan_key = apigateway.CfnUsagePlanKey(self, "WebSocketUsagePlanKey",
            key_id=cfn_api_key.attr_api_key_id,
            key_type="API_KEY",
            usage_plan_id=cfn_usage_plan.attr_id
        )
    
        self.core_api = core_api        