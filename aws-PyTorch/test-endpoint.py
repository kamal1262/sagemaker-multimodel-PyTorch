import json
import boto3


if __name__ == "__main__":
    endpoint_name = "location-recommendation-hostin-0dbc3787d6a74d4188d3a7f9f102968b"
    single_test = json.dumps({"locationIDInput": ["884a1467ffb146b18ae6eda9edd76179"], "count": 5})
    runtime_client = boto3.client('runtime.sagemaker')
    response = runtime_client.invoke_endpoint(EndpointName = endpoint_name,
                                            ContentType = 'application/json',
                                            Body = single_test)
    result = response['Body'].read().decode('ascii')
    print('Predicted label is {}.'.format(result))
