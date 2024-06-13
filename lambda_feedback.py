import os
import json
import boto3
import pandas as pd
import sagemaker    
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role
from io import BytesIO

s3_client = boto3.client('s3')


###input data will be like this 
# { "text": "From: gary@ke4zv.uucp (Gary Coffman) Subject: Re: Math?? (Was US govt & Technolgy Investment Keywords: science? Reply-To: gary@ke4zv.UUCP (Gary Coffman) Organization: Destructive Testing Systems Lines: 23 In article <C71EnF.HJM@ncratl.AtlantaGA.NCR.COM> mwilson@ncratl.AtlantaGA.NCR.COM (Mark Wilson) writes: >In <1993May13.100935.21187@ke4zv.uucp> gary@ke4zv.uucp (Gary Coffman) writes: >|It is, however, now somewhat of an experimental science with the exploration >|of fractals, strange attractors, and artificial life. Whether important >|insights will be gained from these experiments is unknown, but it does >|tend to change the shape of what has mostly been viewed as an abstract >|deductive field. >How do you do experiments in mathematics? Nowadays, usually with a computer. No theory predicted the numeric discoveries listed above. No one can yet write an algorithm that will predict the precise behavior of any of these at any precise level of their evolution. So it remains for experimenters to gather data on their behavior. Gary -- Gary Coffman KE4ZV | You make it, | gatech!wa4mei!ke4zv!gary Destructive Testing Systems | we break it. | uunet!rsiatl!ke4zv!gary 534 Shannon Way | Guaranteed! | emory!kd4nc!ke4zv!gary  Lawrenceville, GA 30244 | ", 
# "label": ["sci", "space"]}


target_list=['autos',
 'baseball',
 'christian',
 'comp',
 'crypt',
 'electronics',
 'forsale',
 'graphics',
 'hardware',
 'hockey',
 'ibm',
 'mac',
 'med',
 'misc',
 'motorcycles',
 'ms-windows',
 'os',
 'pc',
 'politics',
 'rec',
 'religion',
 'sci',
 'soc',
 'space',
 'sport',
 'sys',
 'talk',
 'windows',
 'x']



def update_dataset(json_data):
    s3_uri = "s3://bert-2/dataset/20ng_bydate_small.tsv"
    s3_bucket = s3_uri.split("/")[2]
    s3_key = "/".join(s3_uri.split("/")[3:])
    response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
    content = response['Body'].read().decode('utf-8')
    df = pd.read_csv(BytesIO(content.encode()), sep="\t")
    df_new = pd.DataFrame(columns=['text']+target_list)
    df_new.loc[0, 'text'] = json_data['text']
    
    # Create columns for each label and set values accordingly
    for label in ['autos', 'baseball', 'christian', 'comp', 'crypt', 'electronics', 'forsale', 'graphics', 'hardware', 'hockey', 'ibm', 'mac', 'med', 'misc', 'motorcycles', 'ms-windows', 'os', 'pc', 'politics', 'rec', 'religion', 'sci', 'soc', 'space', 'sport', 'sys', 'talk', 'windows', 'x']:
        df_new[label] = 1 if label in json_data['label'] else 0
    
    df_concatenated = pd.concat([df, df_new], ignore_index=True)
    csv_buffer = BytesIO()
    df_concatenated.to_csv(csv_buffer, sep="\t", header=True, index=False)
    csv_buffer.seek(0)
    s3_client.put_object(Body=csv_buffer.getvalue(), Bucket=s3_bucket, Key="dataset/20ng_bydate_small.tsv")
    
    
    
    
    
def lambda_handler(event, context):
    
    update_dataset(event)


    role = get_execution_role()
    
    output_path = "s3://" + "bert-2" + "/bert-output"
    
    
    instance_type = "ml.m5.xlarge"
    
    est = PyTorch(
        entry_point="train.py",
        source_dir="code",  # directory of your training script
        role=role,
        framework_version="2.1.0",
        py_version="py310",
        instance_type=instance_type,
        instance_count=1,
        volume_size=8,
        output_path=output_path,
        hyperparameters={"batch-size": 16, "epochs": 1, "learning-rate": 1e-05, },
    )
    
    
    est.fit(wait=False)
    
    
    training_job_name = est.latest_training_job.name
    
    # Prepare the training job name as a text file
    training_job_file_content = f"TrainingJobName: {training_job_name}"
    
    # Upload the training job name to S3

    bucket_name = 'bert-2'
    training_job_file_key = "training-job-name.txt"
    
    s3_client.put_object(
        Bucket=bucket_name,
        Key=training_job_file_key,
        Body=training_job_file_content
    )
    
    
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
        
