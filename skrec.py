import nltk.translate.chrf_score
# Import libraries
import torch
import pathlib as pl
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

#plt.switch_backend('agg')

# del transformer
torch.cuda.empty_cache()

import pandas as pd

def explore_tsv_structure(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t')
    print("TSV file structure:")
    print(df.head())
    print("Columns:")
    print(df.columns)
    print("Number of rows:", len(df))

tsv_file = "/mnt/sharedhome/harahus/Tvorenie_gram_chyb/skrec_low_modifed.csv"
explore_tsv_structure(tsv_file)


import h5py
from torch.utils.data import Dataset
random.seed(42)

class Hdf5Dataset(Dataset):
    """Custom Dataset for loading entries from HDF5 databases"""

    def __init__(self, h5_path, transform=None, num_entries=None, randomized=False):
        self.h5f = h5py.File(h5_path, 'r')
        self.size = self.h5f['labels'].shape[0]
        self.transform = transform
        self.randomized = randomized
        self.max_index = num_entries if num_entries is not None else self.size

        # Chooses an offset for the dataset when using a subset of a Hdf5 file
        if randomized:
            if self.size // self.max_index == 0:
                self.offset = 0
            else:
                self.offset = random.choice(range(0, self.size // self.max_index)) * self.max_index
        else:
            self.offset = 0

    def __getitem__(self, index):
        if index >= self.max_index or index >= self.h5f['input'].shape[0]:
            raise StopIteration

        # Diagnostic prints
        #print(f"Accessing index: {index}, Offset: {self.offset}, Dataset size: {self.h5f['input'].shape[0]}")
        
        try:
            input = self.h5f['input'][self.offset + index].decode('utf-8')
        except IndexError as e:
            print(f"Error at index {index} with offset {self.offset}. Total dataset size: {self.h5f['input'].shape[0]}")
            raise e
        
        label = self.h5f['labels'][self.offset + index].decode('utf-8')
        if self.transform is not None:
            features = self.transform(input)
        return input, label

    def __len__(self):
        return self.max_index

    def reshuffle(self):
        if self.randomized:
            if self.size // self.max_index == 0:
                self.offset = 0
            else:
                self.offset = random.choice(range(0, self.size // self.max_index)) * self.max_index
        else:
            print("Please set randomized=True")





import os
from typing import List
from tqdm import tqdm
from torchtext.data import get_tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


# Configuration parameters
SRC_LANGUAGE = 'incorrect'
TGT_LANGUAGE = 'correct'
MAX_LENGTH = 256
VOCAB_SIZE = 2000  # Was initially set to 20000

TRAIN_SAMPLES = 1000
VALID_SAMPLES = 200

# Place-holders
vocab_transform = {}

token_transform = get_tokenizer("basic_english")
#token_transform = AutoTokenizer.from_pretrained("ApoTro/slovak-t5-small")

# Paths configurationdata_dir = 'data/'
data_dir = 'data/'
vocab_dir = 'vocab/'
train_filename = f'{data_dir}1000000.hdf5'
valid_filename = train_filename
test_filename = train_filename
src_vocab_path = f'{vocab_dir}vocab_50K.pth'
tgt_vocab_path = src_vocab_path  # src and target vocab are the same
checkpoint_folder = data_dir

# Diagnostic checks
print(f"Current working directory: {os.getcwd()}")
print(f"Files in {data_dir}: {os.listdir(data_dir)}")

# Check if data files exist
data_files = [train_filename, valid_filename, test_filename]
for filename in data_files:
    if os.path.exists(filename):
        print(f"Data file {filename} exists!")
    else:
        print(f"Data file {filename} does not exist!")



import torchtext as text
import numpy as np
import torch

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# # Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<UNK>','<PAD>', '<BOS>', '<EOS>']

vocab_transform[SRC_LANGUAGE] = torch.load(src_vocab_path)
vocab_transform[TGT_LANGUAGE] = torch.load(tgt_vocab_path)

#Procedure for pretrained embeddings. Not fitting into memory
# def pretrained_embs(name: str, dim: str,max_vectors: int=None):
#     glove_vectors = text.vocab.GloVe(name=name,dim=dim,max_vectors=max_vectors)
#     glove_vocab = text.vocab.vocab(glove_vectors.stoi)
#     pretrained_embeddings = glove_vectors.vectors
#     glove_vocab.insert_token('<UNK>',UNK_IDX)
#     pretrained_embeddings = torch.cat((torch.mean(pretrained_embeddings,dim=0,keepdims=True),pretrained_embeddings))
#     glove_vocab.insert_token('<PAD>',PAD_IDX)
#     pretrained_embeddings = torch.cat((torch.zeros(1,pretrained_embeddings.shape[1]),pretrained_embeddings))
#     glove_vocab.insert_token('<BOS>',PAD_IDX)
#     pretrained_embeddings = torch.cat((torch.rand(1,pretrained_embeddings.shape[1]),pretrained_embeddings))
#     glove_vocab.insert_token('<EOS>',PAD_IDX)
#     pretrained_embeddings = torch.cat((torch.rand(1,pretrained_embeddings.shape[1]),pretrained_embeddings))
#     glove_vocab.set_default_index(UNK_IDX)
#     return glove_vocab,pretrained_embeddings
# vocab, embeddings = pretrained_embs('42B','100',50000)



from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    #truncate sequences longer than MAX_LENGTH
    if(len(token_ids) > MAX_LENGTH - 2):
        token_ids = token_ids[0:MAX_LENGTH -2]
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform,
                                           vocab_transform[ln],
                                           tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


text = ('Data Maining is awesome!','Data Mining is awesome!')
src,tgt = collate_fn([text])
print(src,tgt)


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer

import math
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size,embedding_weights=None):
        super(TokenEmbedding, self).__init__()
        if embedding_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weights,freeze=True,padding_idx=PAD_IDX)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size)

            # self.embedding.weight.requires_grad =False
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network 
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 100,
                 dropout: float = 0.0,
                 embedding_weights = None):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size,embedding_weights)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size,embedding_weights)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, 
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)
    
    def create_mask(src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        def generate_square_subsequent_mask(sz):
            mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask
        
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

        src_padding_mask = (src == PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    

    def encode(self, src: Tensor, src_mask: Tensor, src_padding_mask: Tensor = None):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask, src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
    


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, repeat_size=-1):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
        
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    
    if repeat_size != -1:
        tgt_mask = tgt_mask.repeat(repeat_size, 1, 1)
        src_mask = src_mask.repeat(repeat_size, 1, 1)
        tgt_padding_mask = tgt_padding_mask.repeat(repeat_size, 1, 1)
        src_padding_mask = src_padding_mask.repeat(repeat_size, 1, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


example_mask = create_mask(src,tgt)
_ = plt.imshow(example_mask[1].cpu(),cmap='hot')
plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


torch.manual_seed(0)

VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE].vocab.itos_) # 50k
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 1024
BATCH_SIZE = 24
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

#To save and retrieve checkpoints
model_name = 'transformer_%dE_%dH_%dF_%d_%d.pt'%(EMB_SIZE,NHEAD,FFN_HID_DIM,NUM_DECODER_LAYERS,NUM_DECODER_LAYERS)

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                 NHEAD, VOCAB_SIZE, VOCAB_SIZE, FFN_HID_DIM)

print(f"The model has {count_parameters(transformer):,} trainable parameters")

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

#transformer = nn.DataParallel(transformer, device_ids=[0,1])
transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.1)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-8)


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = Hdf5Dataset(train_filename,num_entries=TRAIN_SAMPLES,randomized=True)
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)


        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Hdf5Dataset(valid_filename,num_entries=VALID_SAMPLES)
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            
        with torch.no_grad():
            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)


        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


# latest_chekpt_file = f"{pl.Path(checkpoint_folder)}/transformer_1mil_train_512E_8H_1024F_6_6_30_epoch.pt"
# checkpoint = torch.load(latest_chekpt_file)
# transformer.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# print(epoch)


from timeit import default_timer as timer
NUM_EPOCHS = 100

train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS+1, 8*NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    torch.save({
        'epoch': epoch,
        'model_state_dict': transformer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
    }, pl.Path(checkpoint_folder)/model_name)



latest_checkpt_train_losses = [3.230, 2.685, 2.575, 2.512, 2.452, 2.447, 2.413, 2.395, 2.374, 2.364, 2.344, 2.342,\
                               2.336, 2.330, 2.318, 2.310, 2.302, 2.284, 2.295, \
                               2.283, 2.285, 2.275, 2.268, 2.267, 2.261, 2.266, \
                               2.258, 2.263, 2.255, 2.244]
latest_checkpt_val_losses = [2.762, 2.622, 2.551, 2.494, 2.471, 2.440, 2.414, 2.393, 2.376, 2.357, 2.349, 2.350, \
                             2.343, 2.336, 2.306, 2.308, 2.303, 2.301, 2.302, \
                             2.292, 2.291, 2.281, 2.262, 2.265, 2.265, 2.271, \
                             2.270, 2.270, 2.258, 2.259]
len(latest_checkpt_train_losses), len(latest_checkpt_val_losses)
_ = plt.plot(latest_checkpt_train_losses)
_ = plt.plot(latest_checkpt_val_losses)


# function to generate output sequence using greedy algorithm
# input: (input_length,batch_size)
# output: (input_length)
def greedy_decode(model, src, src_mask, src_padding_mask, max_len):
    batch_size = src.shape[1]

    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    src_padding_mask = src_padding_mask.to(DEVICE)
    unfinished_sequences = torch.ones(1,batch_size).to(DEVICE)

    context = transformer.encode(src, src_mask,src_padding_mask).to(DEVICE)
    ys = torch.ones(1, batch_size).fill_(BOS_IDX).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, context, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob,dim=1)

        #Predict only PAD_IDX after EOS_IDX is predicted
        next_word = next_word * unfinished_sequences + (1-unfinished_sequences) * PAD_IDX
        ys = torch.cat([ys,
                        next_word], dim=0)

        # if eos_token was found in one sentence, set sentence to finished
        unfinished_sequences = unfinished_sequences.mul((next_word != EOS_IDX).long())

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0:
            break
    return ys.int()


# actual function to correct input sentence
def correct(src_sentence: str, model: torch.nn.Module):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    src_padding_mask = (torch.zeros(1, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask,src_padding_mask, max_len=num_tokens + 5).flatten()
    return ' '.join([vocab_transform[SRC_LANGUAGE].vocab.itos_[i] for i in tgt_tokens if i not in [PAD_IDX,BOS_IDX,EOS_IDX]])


# checkpoint = torch.load(pl.Path(checkpoint_folder)/model_name)
# transformer.load_state_dict(checkpoint['model_state_dict'])

transformer.eval()

# Pick one in 18M examples
test_iter = Hdf5Dataset(test_filename,num_entries=None)

src,trg = random.choice(test_iter)

print("input: \"",src,"\"")
print("target: \"",trg,"\"")

prediction = correct(src,transformer)

print("prediction: \"%s\""%prediction)

