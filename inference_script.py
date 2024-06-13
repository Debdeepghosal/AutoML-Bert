import torch
import json
import logging
import sys
import os

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

# Hyperparameters
MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-05
THRESHOLD = 0.5 # threshold for the sigmoid

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


try:
    from transformers import BertTokenizer, BertModel
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "transformers"])
    from transformers import BertTokenizer, BertModel
    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def model_fn(model_dir):
    logger.info("Loading model...")
    model = BERTClass()
    
    with open(os.path.join(model_dir , 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f,map_location=torch.device(device)))
    

    model = model.to(device).eval()
    
    logger.info("Model loaded.")
    return model

def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    data = json.loads(request_body)["inputs"]
    return data


def predict_fn(input_data, model):
    logger.info("Tokenizing input data...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    encoded_text = tokenizer.encode_plus(
    input_data,
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=True,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',
    )
     
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    token_type_ids = encoded_text['token_type_ids'].to(device)
    output = model(input_ids, attention_mask, token_type_ids)
     # add sigmoid, for the training sigmoid is in BCEWithLogitsLoss
    output = torch.sigmoid(output).detach().cpu()
    # thresholding at 0.5
    output = output.flatten().round().numpy()

    # Correctly identified the topic of the paper: High energy physics
#     print(f"Title: {raw_text}")
    label=''
    for idx, p in enumerate(output):
        if p==1:
    #         print(f"Label: {target_list[idx]}")
            label=label+' '+target_list[idx]
    return label

def output_fn(predictions, content_type):
    assert content_type == "application/json"
    return json.dumps(predictions)

