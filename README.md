# Disclaimers

**Customers are responsible for making their own independent assessment of the information in this document.**

**This document:**

(a) is for informational purposes only, 

(b) represents current AWS product offerings and practices, which are subject to change without notice, and 

(c) does not create any commitments or assurances from AWS and its affiliates, suppliers or licensors. AWS products or services are provided “as is” without warranties, representations, or conditions of any kind, whether express or implied. The responsibilities and liabilities of AWS to its customers are controlled by AWS agreements, and this document is not part of, nor does it modify, any agreement between AWS and its customers. 

(d) is not to be considered a recommendation or viewpoint of AWS

**Additionally, all prototype code and associated assets should be considered:**

(a) as-is and without warranties

(b) not suitable for production environments

(d) to include shortcuts in order to support rapid prototyping such as, but not limitted to, relaxed authentication and authorization processes

**All work produced is open source. More information can be found in the GitHub repo.**


# AOSS POC

0) Setup your environment for python CDK deploy (venv & installing requirements.txt) and setup your layers (check aoss_poc/lambda/custom_packages/src/README.md)
1) Deploy
2) Drop json in S3.  Note, on first go, drop a single JSON file only, then drop the rest after initial file finishes processing.  Known potential race condition on index creation that doesn't have proper error handling ;-).

*We expect trusted content to be in JSON format. Expected Minimum Json Format*
```
[
    {
        "id":"###",
        "content":"Lorem Ipsum"
    },
]
```

3) Validate ingestion (curl or AOSS Dashboard)
Note: manually add your login role to prinicpals on data policy to access dashboard

4) Perform a search (either curl with x-api-key or through console)

*POST /api/aoss/search/*
```
{
    "user_input":"lorem ipsum"
}
```

# Utils
This includes a collection of utilities.

1) json chunker

Splits a json file int he format of 
```
[
    {
        "id":"###",
        "content":"Lorem Ipsum"
    },
    ...
    n
]
```
into smaller chunk files (default 3).  This is useful to avoid lambda timeouts, and process ingest faster.

2) client_ui & alternate query client

This "simulates" a client ui.  It calls search asynchronously and establishes a websocket -- receives response back via websocket once processing completed on backend.

Alternate query program was used in testing PG Vector compared with OpenSearch and was also used to experiment with different LLMs.  You'll find code examples
of how to conect with different LLMs and uses LangChain with messages format.

# Important Security Note
This needs a strong rewrite of CDK to reduce permission scopes. This is for rapid development prototyping only.

# Notes regarding AOSS
Amazon OpenSearch Service>Serverless: Dashboard

1) Can set OU max to 2 index/2 search for prototyping purposes
2) To use dashboard, go to Data access policies>articles-policy and add your console role to the principals

# Useful AOSS Queries
Grab all doc
```
GET _search
{
  "query": {
    "match_all": {}
  }
}
```

Get Timeseries Aggregates (Missed) top 10
```
GET _search
{
  "size": 0,
  "query": {
    "match_all": {}
  },
  "aggs": {
    "top_queries": {
      "terms": {
        "field": "query.keyword",
        "size": 10,
        "order": {
          "_count": "desc"
        }
      },
      "aggs": {
        "doc_count": {
          "value_count": {
            "field": "_id"
          }
        },
        "top_query_hits": {
          "top_hits": {
            "size": 1,
            "_source": {
              "includes": [
                "query"
              ]
            }
          }
        }
      }
    }
  }
}
```




# CDK Setup & Deploy

You should explore the contents of this project. It demonstrates a CDK app with an instance of a stack (`aoss_poc_stack`)
which contains an Amazon SQS queue that is subscribed to an Amazon SNS topic.

The `cdk.json` file tells the CDK Toolkit how to execute your app.

This project is set up like a standard Python project.  The initialization process also creates
a virtualenv within this project, stored under the .venv directory.  To create the virtualenv
it assumes that there is a `python3` executable in your path with access to the `venv` package.
If for any reason the automatic creation of the virtualenv fails, you can create the virtualenv
manually once the init process completes.

To manually create a virtualenv on MacOS and Linux:

```
$ python -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following
step to activate your virtualenv.

```
$ source .venv/bin/activate
```

If you are a Windows platform, you would activate the virtualenv like this:

```
% .venv\Scripts\activate.bat
```

Once the virtualenv is activated, you can install the required dependencies.

```
$ pip install -r requirements.txt
```

At this point you can now synthesize the CloudFormation template for this code.

```
$ cdk synth
```

You can now begin exploring the source code, contained in the hello directory.
There is also a very trivial test included that can be run like this:

```
$ pytest
```

To add additional dependencies, for example other CDK libraries, just add to
your requirements.txt file and rerun the `pip install -r requirements.txt`
command.

## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation

Enjoy!
