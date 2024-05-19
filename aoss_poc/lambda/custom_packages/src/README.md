Consider creating a separate venv for building our Lambda layer

For the handlers and search lambdas we need to build a zip with libraires
that aren't available by default

1) Make a folder called python and cd into that directory

```
mkdir python
cd python
```

2) run this command to download python libraires locally

`pip install requests-aws4auth opensearch-py boto3 botocore -t .`

3) Numpy is unique and takes a little more work to get the appropriate layer:
https://docs.aws.amazon.com/lambda/latest/dg/python-layers.html#python-layer-manylinux

4) Output of step 2 and step 3 should be nested in a folder called python. Note, your target of `cp -r create_layer/lib python/` from step 3 should be the same python folder you are nesting on.

5) python folder should be zipped and called aoss_poc.zip
```
cd ..
zip -r aoss_poc.zip python
```
6) zip file should be moved ../layers/

 `mv aoss_poc.zip ../layers/.`

At time of prototyping
```
boto3==1.34.53
botocore==1.34.53
opensearch-py==2.4.2
```