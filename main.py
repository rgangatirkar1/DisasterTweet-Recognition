import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('/content/train.csv')
print(data)from transformers import pipeline
from google.colab import userdata
userdata.get('HF_KEY')

# Load a pre-trained model for text classification
classifier = pipeline('text-classification', model='bert-base-uncased')
# Check for missing values
print(data.isnull().sum())

# Check data types
print(data.dtypes)import re
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = text.lower().strip()
    return text

data['text'] = data['text'].apply(clean_text)
print(data.head())
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size = 0.2)
for train_indices, test_indices in split.split(data, data[["target"]]):
  trainingSet = data.loc[train_indices]
  testingSet = data.loc[test_indices]

from datasets import Dataset
train_dataset = Dataset.from_pandas(trainingSet)
test_dataset = Dataset.from_pandas(testingSet)
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length = 64 )

# Tokenizing dataset
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2)
# Rename the column containing labels to 'labels'
if 'target' in tokenized_train_dataset.column_names:
    tokenized_train_dataset = tokenized_train_dataset.rename_column('target', 'labels')
elif 'label' in tokenized_train_dataset.column_names:
    tokenized_train_dataset = tokenized_train_dataset.rename_column('label', 'labels')
else:
    raise ValueError("The training dataset is missing the target or label column.")

if 'target' in tokenized_test_dataset.column_names:
    tokenized_test_dataset = tokenized_test_dataset.rename_column('target', 'labels')
elif 'label' in tokenized_test_dataset.column_names:
    tokenized_test_dataset = tokenized_test_dataset.rename_column('label', 'labels')
else:
    raise ValueError("The evaluation dataset is missing the target or label column.")

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,  # Use the tokenized train dataset
    eval_dataset=tokenized_test_dataset    # Use the tokenized test dataset
)



print(tokenized_train_dataset.column_names)
print(tokenized_test_dataset.column_names)

trainer.train()
trainer.evaluate()

RealTestData = pd.read_csv('/content/test.csv')

# Prepare the dataset for predictions
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=64)

# Convert the DataFrame to a Dataset
test_dataset = Dataset.from_pandas(RealTestData)

# Tokenize the test dataset
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Initialize Trainer with the trained model
trainer = Trainer(model=model)

# Make predictions
predictions = trainer.predict(tokenized_test_dataset).predictions
predicted_labels = predictions.argmax(axis=-1)

# Create the final DataFrame for submission
final_df = pd.DataFrame(RealTestData['id'])  # Assuming 'id' is the identifier column in test.csv
final_df['label'] = predicted_labels  # Assuming your model outputs labels
final_df.rename(columns={'label': 'target'}, inplace=True)
final_df.to_csv('/content/predictions.csv', index=False)
print("File saved successfully")
