from transformers import AutoTokenizer
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict
import math # for perplexity evaluation
from huggingface_hub import notebook_login
from transformers import TrainingArguments
from transformers import Trainer
   
def tokenize_function(batch):
    result = tokenizer(batch['text'], padding="max_length", truncation=True, max_length=chunk_size, return_tensors="pt")
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

if not torch.cuda.is_available():
    print("CUDA not available")
    quit()
    
# path to UCSB text files with the speeches
path = './DataUCSB/'
   
print("Load pretrained model")
model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
# downloads model, about 256MB

# distilbert trains faster than vanilla bert with little loss in downstream performance, apparently
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Trainer is spitting out memory errors, will try 64 instead of 128, see if it works
#chunk_size = 128 # smaller than model_max_length of 512, for memory considerations
chunk_size = 32

print("Load dataset")
dataset = load_dataset(path)

train_testvalid = dataset["train"].train_test_split(seed = 33, test_size=0.2)
test_valid = train_testvalid["test"].train_test_split(seed = 33, test_size=0.5)
oba_data = DatasetDict({
    "train": train_testvalid["train"],
    "test": test_valid["test"],
    "valid": test_valid["train"]
})

print("Tokenize the dataset")
# Use batched=True to activate fast multithreading!
tokenized_datasets = oba_data.map(
    tokenize_function, batched=True, remove_columns=["text"]
)

# Mask 15% of the tokens
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

print("Group into lm_datasets")
# group data and add labels
lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# Don't need to downsample, so just copy
downsampled_dataset = lm_datasets

# Login to hugginface to do training
# Run the following "in your favorite terminal and log in there."
# hugginface-cli login

batch_size = chunk_size
# Show the training loss with every epoch
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

print("Set training arguments")
training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-speeches",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=True,
    fp16=True,
    logging_steps=logging_steps,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
print("Start trainer.train()")
trainer.train()

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

print("Training complete, pushing to hub")
trainer.push_to_hub()