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

3) Output should be nested in a folder called python

4) python folder should be zipped and called aoss_poc.zip
```
cd ..
zip -r aoss_poc.zip python
```
5) zip file should be moved ../layers/

 `mv aoss_poc.zip ../layers/.`

At time of prototyping
```
boto3==1.34.53
botocore==1.34.53
opensearch-py==2.4.2
```