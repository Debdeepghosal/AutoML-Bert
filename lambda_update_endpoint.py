import json
import os

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role, Session
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

endpoint_name = os.environ['ENDPOINT_NAME']


def lambda_handler(event, context):
    
    s3_uri = "s3://bert-2/training-job-name.txt"
    
    # Extract bucket name and key from the S3 URI
    s3_bucket = s3_uri.split("/")[2]
    s3_key = "/".join(s3_uri.split("/")[3:])
    
    # Read the training job name from the S3 bucket
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
    file_content = response['Body'].read().decode('utf-8').strip()
    
    # Extract the training job name from the file content
    training_job_name = None
    for line in file_content.split('\n'):
        if line.startswith("TrainingJobName:"):
            training_job_name = line.split(":")[1].strip()
            break
    
    if not training_job_name:
        return {
            'statusCode': 500,
            'body': json.dumps('TrainingJobName not found in the file.')
        }
    

    sagemaker_client = boto3.client('sagemaker')
    training_info = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
    model_artifact = training_info['ModelArtifacts']['S3ModelArtifacts']
    
    try:
        response = sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint '{endpoint_name}' deleted successfully.")
        
        response = sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"Endpoint configuration '{endpoint_name}' deleted successfully.")
        
    except Exception as e:
        print(f"Error deleting endpoint '{endpoint_name}': {e}")
        

    role = get_execution_role()
    
    
    model = PyTorchModel(
        entry_point="inference.py",
        source_dir="code",
        role=role,
        model_data=model_artifact,
        framework_version="2.1.0",
        py_version="py310",
    )

    instance_type = "ml.m5.large"
    
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        endpoint_name=endpoint_name,
        wait=False
    )
        

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }

