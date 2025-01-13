import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelWithLMHead
import csv

# load the dataset
with open('/home/yaning/Documents/LLM/E5/extracted.csv', mode='r', newline='') as file:
    reader = csv.reader(file)
    
    for r in reader:
        texts = r

texts = texts[:3]

# pre trained model and tokenizer
model_name = "dbmdz/german-gpt2"
tokenizer = AutoTokenizer.from_pretrained("dbmdz/german-gpt2")
model = AutoModelWithLMHead.from_pretrained(model_name)

# Ensure padding tokens are correctly added if missing
tokenizer.pad_token = tokenizer.eos_token



# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)

# Convert text to a Dataset
dataset = Dataset.from_dict({"text": texts})

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# model.to("cpu")
tokenized_dataset = tokenized_dataset.map(lambda x: {
    'input_ids': torch.tensor(x['input_ids']),
    'attention_mask': torch.tensor(x['attention_mask']),
}, batched=True)

# Step 3: Prepare Data Collator for Language Modeling
# This collator dynamically pads the input and is needed for GPT-2 language modeling
# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False,  # GPT-2 is causal language modeling, not masked
# )


# Step 4: Set up TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory where the model checkpoints will be saved
    overwrite_output_dir=True,
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=2,  # Batch size per device (GPU)
    per_device_eval_batch_size=2,  # Eval batch size
    logging_dir='./logs',  # Log directory
    logging_steps=10,
    save_steps=500,  # Save model checkpoint every 500 steps
    prediction_loss_only=True,
)


# tokenized_dataset.to("cpu")
# training_args.to("cpu")
# Step 5: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    # data_collator=data_collator,
)

# Step 6: Train the model
trainer.train()
