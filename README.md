# 🤖 AutoML Implementation on BERT Classifier

## Author: Debdeep Ghosal

---

## 🗺️ Roadmap

1. **📂 Dataset Collection**
2. **🧹 Data Preprocessing**
3. **🧠 Initial Model Training**
4. **🚀 Model Deployment on Sagemaker Endpoint**
5. **🔌 Set up Model Inference API**
6. **🤖 AutoML Implementation**

---

### 📂 Dataset Collection

- The dataset has been collected from [20 Newsgroups Dataset](http://qwone.com/~jason/20Newsgroups/).
- It contains 18,846 documents across 20 newsgroups with multiple labels (e.g., baseball, comp, crypt, talk, religion, etc.).

---

### 🧹 Data Preprocessing

- The dataset is loaded in the Kaggle environment.
- The directory-structured dataset corresponding to the classes is inserted into a pandas DataFrame.
- The dataset is randomly shuffled, and categorical encoding is applied to the labels.
- Categories with low frequency and empty rows are removed.
- The cleaned DataFrame is exported for model training.

---

### 🧠 Initial Model Training

The initial model is trained on a free Kaggle GPU. The following steps are involved:

1. Load required libraries and dataset.
2. Split dataset into train, test, and validation sets; load the encoder.
3. Create a custom dataset class for tokenized data points, and create train, test, and validation dataset instances with this class.
4. Create respective dataloaders for the dataset instances.
5. Instantiate a `BERTClass` that loads the `bert-base-uncased` model.
6. Define loss function, optimizer, and hyperparameters.
7. Create and run training and evaluation functions for a set number of epochs, saving model weights upon improvement.

---

### 🚀 Model Deployment on Sagemaker Endpoint

1. Create a Sagemaker notebook instance; upload the trained model weights to an S3 bucket.
2. Create an inference script defining:
   - Model loading
   - Input processing
   - Prediction making
   - Output sending
3. Create a `PyTorchModel` instance, passing the inference script and model weight (S3 URI).
4. Deploy the model, choosing an instance type.

---

### 🔌 Set up Model Inference API

1. Create a Lambda function with permissions to invoke a Sagemaker endpoint and get predictions.
2. Attach the Lambda function to an API Gateway to receive data in JSON format and respond with predictions.

---

### 🤖 AutoML Implementation

1. Create a Lambda function and attach it to an API Gateway.
2. The API receives feedback data, which is:
   - Cleaned, preprocessed, merged with the previous dataset, and saved in the S3 bucket.
   - Used to start a new training job with the updated dataset.
3. Set up an EventBridge rule to trigger a Lambda function upon training job completion, updating the current endpoint with new weights.
4. Maintain endpoint name consistency for seamless integration.

---

## 🚀 Future Scope and Improvements

- Hyperparameter tuning (not implemented due to large compute requirements).
- Using a secondary endpoint to ensure ~100% uptime.
- Setting up events to train the model during low traffic periods using CloudWatch and EventBridge.

---

## 📌 Author’s Note

- Each Lambda function requires specific policies attached to its role to interact with other AWS services (e.g., IAM, Sagemaker, S3).
- The Lambda function associated with training the model also requires the training script (available in this GitHub repository).
- The name of the scheduled training job is saved in S3 and used by the BERT-update Lambda function during endpoint deployment.

---

Thank you for exploring this project! For any queries or contributions, feel free to reach out.

---

