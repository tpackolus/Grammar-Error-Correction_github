#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install datasets tqdm pandas


# In[ ]:


#pip install transformers[torch]

# In[ ]:


#!pip install sentencepiece

# In[ ]:


#!pip install transformers

# In[ ]:


#!pip install errant

# In[ ]:


#!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# In[ ]:


#!pip install datasets

# In[ ]:


#!pip install wandb

# In[ ]:


#! pip install transformers

# In[ ]:


#!pip install evaluate


# In[ ]:


#!pip install nltk

# In[ ]:


# import errant
# import spacy

# nlp = spacy.load('en')
# annotator = errant.load('en', nlp)

# In[ ]:


import torch
torch.cuda.empty_cache()
import gc

gc.collect()

# In[ ]:


import warnings
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.")
import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

# In[ ]:


import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# In[ ]:


!nvidia-smi

# In[ ]:


import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

# In[ ]:


import random
import numpy as np
import torch
import datasets

# In[ ]:


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

set_seed(42)

# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"]="0,"
#torch.cuda.set_device(0)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # get them thangs ready

# In[ ]:


pd.set_option('display.max_colwidth', None)

# In[ ]:


!gdown "1aQSSoL7bCwBpCIWeiS22S97tsDmMajbB"

# In[ ]:


column_names = ['labels', 'input']
df = pd.read_csv('5000000.csv', delimiter='\t', names=column_names, header=None)
print(df.shape)


# In[ ]:


df.head()

# In[ ]:


df['input'] = df['input'].str[8:-6]
df['labels'] = df['labels'].str[8:-6]


# In[ ]:


df.head()


# In[ ]:


from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
  )

from torch.utils.data import Dataset, DataLoader

# In[ ]:


# model_name = 'ApoTro/slovak-t5-small'
# tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512)
# model = T5ForConditionalGeneration.from_pretrained(model_name)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("ApoTro/slovak-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("ApoTro/slovak-t5-small")

# In[ ]:


def calc_token_len(example):
    return len(tokenizer(example).input_ids)

# In[ ]:


df = df.sample(n=df.shape[0] // 8)
df.shape

# In[ ]:


from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.1, shuffle=True)
train_df.shape, test_df.shape

# In[ ]:


test_df['input_token_len'] = test_df['input'].apply(calc_token_len)

# In[ ]:


test_df.head()

# In[ ]:


test_df["input_token_len"].describe()

# In[ ]:


from datasets import Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# In[ ]:


test_dataset

# In[ ]:


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


# In[ ]:


from transformers import TrainerCallback

class CustomLoggingCallback(TrainerCallback):
    def __init__(self, log_filename):
        super().__init__()
        self.log_filename = log_filename

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Assuming metrics contain your desired logging information. Adjust as necessary.
        with open(self.log_filename, 'a') as f:
            log_str = f"{state.global_step}\t"
            for key, value in metrics.items():
                log_str += f"{value:.6f}\t"
            log_str += "\n"
            f.write(log_str)

        print(log_str, end='')  # This line will print the metrics to the console.


# In[ ]:


dataset =GrammarDatasetWithLimitSkip(test_dataset, tokenizer, True)
print(dataset[121])

# In[ ]:


!pip install rouge_score

# In[ ]:


from datasets import load_metric
rouge_metric = load_metric("rouge")


# In[ ]:


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest', return_tensors='pt')

# In[ ]:


from transformers import TrainingArguments, Trainer

# Assuming CustomLoggingCallback is previously defined
log_callback = CustomLoggingCallback(log_filename="logs.txt")

batch_size = 24

args = Seq2SeqTrainingArguments(
    output_dir="./data/c4_200m/weights",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Added this line
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=3e-5,
    num_train_epochs=100,
    weight_decay=0.03,
    save_total_limit=2,
    logging_steps=10,
    predict_with_generate=True,
    gradient_accumulation_steps=6,
    load_best_model_at_end=True,
    logging_dir="./logs",
    report_to="wandb"
)


# In[ ]:


import numpy as np
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate import gleu_score

from rouge_score import rouge_scorer, scoring
import evaluate

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


# In[ ]:




# In[ ]:


trainer = Seq2SeqTrainer(model=model,
                args=args,
                train_dataset=GrammarDatasetWithLimitSkip(train_dataset, tokenizer),
                eval_dataset=GrammarDatasetWithLimitSkip(test_dataset, tokenizer),
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[log_callback],
                compute_metrics=compute_metrics)

# In[ ]:


print("Starting training...")
trainer.train()
print("Training complete.")

print("Evaluating...")
results = trainer.evaluate()
print("Evaluation results:", results)



# In[ ]:


# # Capture the previous cell's output
# output = _ 

# # Save to a .txt file
# with open("results.txt", "w") as file:
#     file.write(str(output))

# print("Results saved to results.txt")


# In[ ]:


#trainer.save_model('t5_gec_model')

# In[ ]:


#!zip -r 't5_gec_model.zip' 't5_gec_model'

# In[ ]:


#!rsync -rhPu --info=progress2 "t5_gec_model/t5_gec_model.zip" "/content/drive/MyDrive/c4_200m"

# In[ ]:


# import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration

# In[ ]:


# model_name = 't5_gec_model'
# torch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name).to(torch_device)

# def correct_grammar(input_text,num_return_sequences):
#   batch = tokenizer([input_text],truncation=True,padding='max_length',max_length=64, return_tensors="pt").to(torch_device)
#   translated = model.generate(**batch,max_length=64,num_beams=4, num_return_sequences=num_return_sequences, temperature=1.5)
#   tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
#   return tgt_text

# In[ ]:


# text = 'Ahoj ako sa mas'
# print(correct_grammar(text, num_return_sequences=1))

# In[ ]:



