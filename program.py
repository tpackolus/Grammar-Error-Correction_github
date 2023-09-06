
import torch
import gc
import warnings
import os
import pandas as pd
import argparse
import glob
import json
import time
import logging
import re
import nltk
import random
import numpy as np
import evaluate
from itertools import chain
from string import punctuation
import datasets
from datasets import load_dataset
from datasets import Dataset
from datasets import load_metric
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate import gleu_score
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset, DataLoader
from rouge_score import rouge_scorer, scoring
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq,
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    TrainerCallback,
    TrainingArguments, 
    Trainer
)







os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
gc.collect()
torch.cuda.empty_cache()

warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.")
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
pd.set_option('display.max_colwidth', None)

nltk.download('punkt')

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

set_seed(42)

column_names = ['labels', 'input']
df = pd.read_csv('/mnt/sharedhome/harahus/Tvorenie_gram_chyb/daniel_data/50000000.csv', delimiter='\t', names=column_names, header=None)
                                                                
print(df.shape)
df.head()
df['input'] = df['input'].str[8:-6]
df['labels'] = df['labels'].str[8:-6] 
df.head()

tokenizer = AutoTokenizer.from_pretrained("ApoTro/slovak-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("ApoTro/slovak-t5-small")

def calc_token_len(example):
    return len(tokenizer(example).input_ids)

  
df = df.sample(n=df.shape[0] // 8)
df.shape

train_df, test_df = train_test_split(df, test_size=0.1, shuffle=True)
train_df.shape, test_df.shape
test_df['input_token_len'] = test_df['input'].apply(calc_token_len) 
test_df.head()  
test_df["input_token_len"].describe()

from datasets import Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
test_dataset

from torch.utils.data import Dataset, DataLoader
class GrammarDatasetWithLimitSkip(Dataset):
    def __init__(self, dataset, tokenizer, print_text=False, max_skips=10):
        self.dataset = dataset
        self.pad_to_max_length = False
        self.tokenizer = tokenizer
        self.print_text = print_text
        self.max_len = 64
        self.max_skips = max_skips

    def __len__(self):
        return len(self.dataset)

    def tokenize_data(self, example):
        input_, target_ = example.get('input', None), example.get('labels', None)

        # Tokenize the input and target
        tokenized_inputs = self.tokenizer(input_, pad_to_max_length=self.pad_to_max_length,
                                          max_length=self.max_len, return_attention_mask=True)

        tokenized_targets = self.tokenizer(target_, pad_to_max_length=self.pad_to_max_length,
                                           max_length=self.max_len, return_attention_mask=True)

        inputs = {
            "input_ids": tokenized_inputs['input_ids'],
            "attention_mask": tokenized_inputs['attention_mask'],
            "labels": tokenized_targets['input_ids']
        }

        return inputs

    def __getitem__(self, index):
        skip_count = 0
        # Keep fetching the next example until a valid one is found or skip limit is reached
        while skip_count < self.max_skips:
            example = self.dataset[index]
            input_, target_ = example.get('input', None), example.get('labels', None)
            
            # Check if either input or labels is None
            if input_ is None or target_ is None:
                print(f"Warning: Skipping problematic data at index {index}")
                index += 1  # Move to the next example
                skip_count += 1
                continue
            
            # If valid example found, break out of the loop
            break

        # If maximum skip limit reached, raise an error
        if skip_count == self.max_skips:
            raise ValueError(f"Reached maximum skip limit of {self.max_skips} at index {index}")

        inputs = self.tokenize_data(example)

        if self.print_text:
            for k in inputs.keys():
                print(k, len(inputs[k]))

        return inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Splitting for ROUGE
    decoded_preds_split = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels_split = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Initialize RougeScorer object
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Compute ROUGE
    scores = [scorer.score(target, prediction) for target, prediction in zip(decoded_labels_split, decoded_preds_split)]
    rouge1_f = np.mean([score['rouge1'].fmeasure for score in scores]) * 100
    rouge2_f = np.mean([score['rouge2'].fmeasure for score in scores]) * 100
    rougel_f = np.mean([score['rougeL'].fmeasure for score in scores]) * 100
    
    result = {
        "rouge-1": rouge1_f,
        "rouge-2": rouge2_f,
        "rouge-L": rougel_f
    }
    
    # Compute BLEU and GLEU with Smoothing for BLEU
    smooth = SmoothingFunction().method1
    bleu_scores = [sentence_bleu([ref], pred, smoothing_function=smooth) for ref, pred in zip(decoded_labels, decoded_preds)]
    gleu_scores = [gleu_score.sentence_gleu([ref], pred) for ref, pred in zip(decoded_labels, decoded_preds)]
    
    result["bleu"] = np.mean(bleu_scores) * 100
    result["gleu"] = np.mean(gleu_scores) * 100
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


class CustomLoggingCallback(TrainerCallback):
    def __init__(self, log_filename):
        super().__init__()
        self.log_filename = log_filename

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        with open(self.log_filename, 'a') as f:
            log_str = f"{state.global_step}\t"
            for key, value in metrics.items():
                log_str += f"{value:.6f}\t"
            log_str += "\n"
            f.write(log_str)

dataset =GrammarDatasetWithLimitSkip(test_dataset, tokenizer, True)
print(dataset[121])

rouge_metric = load_metric("rouge") 
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest', return_tensors='pt')

log_callback = CustomLoggingCallback(log_filename="logs.txt")

batch_size = 24


args = Seq2SeqTrainingArguments(
    output_dir="./data/c4_200m/weights",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Added this line
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=3e-5,
    num_train_epochs=15,
    weight_decay=0.03,
    save_total_limit=2,
    logging_steps=10,
    predict_with_generate=True,
    gradient_accumulation_steps=6,
    load_best_model_at_end=True,
    logging_dir="./logs",
    report_to="wandb"
)

trainer = Seq2SeqTrainer(model=model,
                args=args,
                train_dataset=GrammarDatasetWithLimitSkip(train_dataset, tokenizer),
                eval_dataset=GrammarDatasetWithLimitSkip(test_dataset, tokenizer),
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[log_callback],
                compute_metrics=compute_metrics)


trainer.train()
trainer.save_model('t5_gec_model_50000000')