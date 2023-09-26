import os
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from collections import Counter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np

# 1. Basic Tokenizer
class BasicTokenizer:
    def tokenize(self, text):
        return text.split()

    def encode(self, text, vocab):
        tokens = [vocab.get(word, vocab['<UNK>']) for word in self.tokenize(text)]
        tokens.append(vocab['<EOS>'])
        return tokens

    def decode(self, tokens, reverse_vocab):
        return ' '.join([reverse_vocab[token] for token in tokens if token not in [vocab['<EOS>'], vocab['<PAD>']]])

# 2. Data Loading and Preprocessing
class GrammarCorrectionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, vocab):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        source_text, target_text = row['incorrect'], row['correct']
        source_tokens = self.tokenizer.encode(source_text, self.vocab)
        target_tokens = self.tokenizer.encode(target_text, self.vocab)
        return torch.tensor(source_tokens), torch.tensor(target_tokens)

def build_vocab(data, tokenizer):
    counter = Counter()
    for sentence in data:
        counter.update(tokenizer.tokenize(sentence))
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for word, _ in counter.most_common():
        vocab[word] = len(vocab)
    return vocab

def collate_fn(batch):
    src, tgt = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=0)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=0)
    return src, tgt

data_sample = pd.read_csv('/mnt/sharedhome/harahus/Tvorenie_gram_chyb/skrec_1000000.csv', sep='\t')
tokenizer = BasicTokenizer()
vocab = build_vocab(pd.concat([data_sample['incorrect'], data_sample['correct']]), tokenizer)
reverse_vocab = {v: k for k, v in vocab.items()}

# Splitting the dataset
total_size = len(data_sample)
train_size = int(0.8 * total_size)
test_size = int(0.1 * total_size)
val_size = total_size - train_size - test_size

train_data, test_data, val_data = random_split(data_sample, [train_size, test_size, val_size])

train_dataset = GrammarCorrectionDataset(train_data.dataset.iloc[train_data.indices], tokenizer, vocab)
test_dataset = GrammarCorrectionDataset(test_data.dataset.iloc[test_data.indices], tokenizer, vocab)
val_dataset = GrammarCorrectionDataset(val_data.dataset.iloc[val_data.indices], tokenizer, vocab)

BATCH_SIZE = 1 * torch.cuda.device_count()
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# 3. Model Definition
class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, nhead, dropout=0.5, pad_idx=0):
        super(Seq2SeqTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        
        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # LSTM Decoder
        self.decoder = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.max_length = 60

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        memory = self.transformer_encoder(src)
        outputs, _ = self.decoder(tgt)
        return self.fc(outputs)

# Model, Optimizer, Criterion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Seq2SeqTransformer(len(vocab), 8, 8, 2, 4).to(device)

# Use DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
gradient_accumulation_steps = 16

# Definícia funkcie pre trénovanie na časti dát
def train_on_chunk(chunk_data, model, optimizer, criterion, device):
    dataset = GrammarCorrectionDataset(chunk_data, tokenizer, vocab)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0
    for batch, (src, tgt) in tqdm(enumerate(loader), total=len(loader)):
        src, tgt = src.to(device), tgt.to(device)
        outputs = model(src, tgt)
        outputs = outputs.view(-1, outputs.size(2))
        tgt = tgt.view(-1)
        loss = criterion(outputs, tgt)
        
        if torch.isnan(loss):
            print("NaN loss encountered. Skipping optimization for this batch.")
            continue
        
        loss.backward()

        if (batch + 1) % gradient_accumulation_steps == 0:
            clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        del src, tgt, outputs

    torch.cuda.empty_cache()

    average_loss = total_loss / (len(loader) / gradient_accumulation_steps)
    return average_loss

# Definícia funkcie pre krokové trénovanie
def step_training(data_sample, num_epochs=10, model_path="model_checkpoint.pth"):
    chunk_size = 5000
    num_chunks = len(data_sample) // chunk_size

    for epoch in range(num_epochs):  # Epoch loop
        print(f"Starting Epoch {epoch + 1}/{num_epochs}")
        for i in range(num_chunks):
            print(f"Training on chunk {i+1}/{num_chunks}")
        
            # Load model from checkpoint if it exists
            if i != 0 and os.path.exists(model_path):
                if torch.cuda.device_count() > 1:
                    model.module.load_state_dict(torch.load(model_path))
                else:
                    model.load_state_dict(torch.load(model_path))
        
            chunk_data = data_sample.iloc[i*chunk_size : (i+1)*chunk_size]
            loss = train_on_chunk(chunk_data, model, optimizer, criterion, device)
            
            # Save the model checkpoint
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
        
            print(f"Chunk {i+1} completed. Loss: {loss:.4f}")
            print("-" * 50)

# Krokové trénovanie
step_training(data_sample)

# 5. Evaluation
def compute_accuracy(loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            outputs = model(src, tgt)
            predicted = outputs.argmax(2)
            total += tgt.size(0) * tgt.size(1)
            correct += (predicted == tgt).sum().item()
    return correct / total

train_acc = compute_accuracy(train_loader)
test_acc = compute_accuracy(test_loader)
val_acc = compute_accuracy(val_loader)

print(f"\nTraining Accuracy: {train_acc*100:.2f}%")
print(f"Testing Accuracy: {test_acc*100:.2f}%")
print(f"Validation Accuracy: {val_acc*100:.2f}%")

def beam_search_decoder(predictions, top_k=3, max_length=50):
    sequences = [[list(), 0.0]]
    
    for row in predictions:
        all_candidates = []
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score - np.log(row[j])]
                all_candidates.append(candidate)
        
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        sequences = ordered[:top_k]
        
    return sequences[0][0]

def correct_sentence_with_beam_search(sentence, beam_width=3, max_length=60):
    model.eval()
    tokens = tokenizer.encode(sentence, vocab)
    src = torch.tensor(tokens).unsqueeze(0).to(device)
    tgt_init = torch.tensor([vocab['<SOS>']]).unsqueeze(0).to(device)
    
    all_predictions = []
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(src, tgt_init)
            predictions = torch.nn.functional.softmax(outputs, dim=2)
            all_predictions.append(predictions[0, -1].cpu().numpy())
        
        predicted = beam_search_decoder(all_predictions, top_k=beam_width)
        tgt_init = torch.tensor(predicted).unsqueeze(0).to(device)
        
        if predicted[-1] == vocab['<EOS>']:
            break

    return tokenizer.decode(predicted, reverse_vocab)

# Test beam search correction
for idx in range(10):
    incorrect_example = test_data.dataset['incorrect'].iloc[test_data.indices[idx]]
    correct_example = test_data.dataset['correct'].iloc[test_data.indices[idx]]
    predicted_sentence = correct_sentence_with_beam_search(incorrect_example)

    print("\nGiven Incorrect Sentence:")
    print(incorrect_example)
    print("\nExpected Correct Sentence:")
    print(correct_example)
    print("\nModel's Correction:")
    print(predicted_sentence)
    print("-" * 40)
