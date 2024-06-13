
import sys
import subprocess
import argparse
import logging
import json

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages to install
packages = ["transformers", "pandas","scikit-learn","boto3"]

# Install each package
for package in packages:
    print(f"Installing {package}...")
    install(package)
    print(f"{package} installed successfully!")

import os
import pandas as pd
import numpy as np
import shutil
from collections import defaultdict
import torch
import torch.nn as nn
from transformers import AdamW
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import boto3

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len, target_list):
        self.tokenizer = tokenizer
        self.df = df
        self.title = list(df['text'])
        self.targets = self.df[target_list].values
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index]),
            'title': title
        }


def data_loading():
    # s3_uri = 's3://bert-2/dataset/20ng_bydate.tsv'
    s3_uri = 's3://bert-2/dataset/20ng_bydate_small.tsv'
    data_dir = "/dataset/"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    s3 = boto3.client('s3')
    
    bucket, key = s3_uri.split('//')[1].split('/', 1)

    # Download the file from S3 to local
    local_file_path = os.path.join(data_dir, '20ng_bydate.tsv')
    s3.download_file(bucket, key, local_file_path)

    # Read the TSV file using pandas
    df_data = pd.read_csv(local_file_path, sep="\t")

    
#     df_data = pd.read_csv(os.path.join(data_dir,"20ng_bydate.tsv"), sep="\t")
    
    # split into train and test
    df_train, df_test = train_test_split(df_data, random_state=77, test_size=0.30, shuffle=True)
    # split test into test and validation datasets
    df_test, df_valid = train_test_split(df_test, random_state=88, test_size=0.50, shuffle=True)
    target_list = list(df_data.columns)
    target_list = target_list[1:]
    train_dataset = CustomDataset(df_train, tokenizer, MAX_LEN, target_list)
    valid_dataset = CustomDataset(df_valid, tokenizer, MAX_LEN, target_list)
    test_dataset = CustomDataset(df_test, tokenizer, MAX_LEN, target_list)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, 
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_data_loader = torch.utils.data.DataLoader(valid_dataset, 
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_data_loader = torch.utils.data.DataLoader(test_dataset, 
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    return train_data_loader,val_data_loader,test_data_loader,target_list






class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, len(target_list))

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

    
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)



def train_model(training_loader, model, optimizer):

    losses = []
    correct_predictions = 0
    num_samples = 0
    # set model to training mode (activate droput, batch norm)
    model.train()
    # initialize the progress bar
    training_loader_length = len(training_loader)

    for batch_idx, data in enumerate(training_loader):
        # Calculate progress percentage
        progress_percent = (batch_idx + 1) / training_loader_length * 100

        # Print progress
        print(f"Progress: {progress_percent:.2f}% ({batch_idx + 1}/{training_loader_length})", end='\r')

        # Process your data
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        # forward
        outputs = model(ids, mask, token_type_ids) # (batch,predict)=(32,8)
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())
        # training accuracy, apply sigmoid, round (apply thresh 0.5)
        outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
        targets = targets.cpu().detach().numpy()
        correct_predictions += np.sum(outputs==targets)
        num_samples += targets.size   # total number of elements in the 2D array

        # backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # grad descent step
        optimizer.step()

        # Update progress bar
        #loop.set_description(f"")
        #loop.set_postfix(batch_loss=loss)

    # returning: trained model, model accuracy, mean loss
    return model, float(correct_predictions)/num_samples, np.mean(losses)


def eval_model(validation_loader, model, optimizer):
    losses = []
    correct_predictions = 0
    num_samples = 0
    # set model to eval mode (turn off dropout, fix batch norm)
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader, 0):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

            # validation accuracy
            # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
            outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
            targets = targets.cpu().detach().numpy()
            correct_predictions += np.sum(outputs==targets)
            num_samples += targets.size   # total number of elements in the 2D array

    return float(correct_predictions)/num_samples, np.mean(losses)
     
    
    
    
    
def training():
    
    model = BERTClass()

    model.to(device)
    # define the optimizer
    optimizer = AdamW(model.parameters(), lr = 1e-5)    
    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(1, EPOCHS+1):
        print(f'Epoch {epoch}/{EPOCHS}')
        model, train_acc, train_loss = train_model(train_data_loader, model, optimizer)
        val_acc, val_loss = eval_model(val_data_loader, model, optimizer)

        print(f'train_loss={train_loss:.4f}, val_loss={val_loss:.4f} train_acc={train_acc:.4f}, val_acc={val_acc:.4f}')

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        # save the best model
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), os.path.join(args.model_dir,"model.pth"))
            best_accuracy = val_acc


            
            
def parse_args():
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for testing (default: 16)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-05,
        metavar="LR",
        help="learning rate (default: 1e-05)",
    )
    

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    return parser.parse_args()            
            
            
            
            
            
            
            
            
if __name__ == "__main__":
    
    args = parse_args()
    
    # Hyperparameters
    MAX_LEN = 256
    TRAIN_BATCH_SIZE = args.batch_size
    VALID_BATCH_SIZE = args.test_batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    THRESHOLD = 0.5 # threshold for the sigmoid
    
    
    train_data_loader,val_data_loader,test_data_loader,target_list=data_loading()

    
    training()            

