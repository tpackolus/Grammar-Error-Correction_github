import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import math

# 1. DATA PREPROCESSING

# Load data from the provided CSV file
df = pd.read_csv('/mnt/sharedhome/harahus/Tvorenie_gram_chyb/skrec_low_modifed.csv', sep='\t')

# Tokenization and Encoding
word2index = {"<pad>": 0, "<unk>": 1}
index2word = {0: "<pad>", 1: "<unk>"}

def tokenize_and_encode(sentence):
    tokens = sentence.split()
    encoded = [word2index.get(token, word2index["<unk>"]) for token in tokens]
    for token in tokens:
        if token not in word2index:
            word2index[token] = len(word2index)
            index2word[len(word2index)] = token
    return encoded

# Check if 'incorrect_encoded' and 'correct_encoded' exist in df and create them if not
if 'incorrect_encoded' not in df.columns:
    df['incorrect_encoded'] = df['incorrect'].apply(tokenize_and_encode)
else:
    df['incorrect_encoded'] = df['incorrect_encoded'].apply(ast.literal_eval)  # Convert strings back to lists
    
if 'correct_encoded' not in df.columns:
    df['correct_encoded'] = df['correct'].apply(tokenize_and_encode)
else:
    df['correct_encoded'] = df['correct_encoded'].apply(ast.literal_eval)  # Convert strings back to lists

# Padding
def pad_sequences(sequences, pad_value=0):
    max_len = max(map(len, sequences))
    return [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]

df['incorrect_encoded'] = pad_sequences(df['incorrect_encoded'].tolist())
df['correct_encoded'] = pad_sequences(df['correct_encoded'].tolist())

# Maximum sequence length
max_seq_len = max(max(map(len, df['incorrect_encoded'])), max(map(len, df['correct_encoded'])))

# Splitting the data
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

# Convert dataframes to DataLoader
def df_to_loader(dataframe, batch_size):
    src_data = torch.tensor(dataframe['incorrect_encoded'].tolist())
    tgt_data = torch.tensor(dataframe['correct_encoded'].tolist())
    dataset = TensorDataset(src_data, tgt_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

BATCH_SIZE = 2
train_loader = df_to_loader(train_df, BATCH_SIZE)
val_loader = df_to_loader(val_df, BATCH_SIZE)
test_loader = df_to_loader(test_df, BATCH_SIZE)
# 2. MODEL DEFINITION

class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, device):
        super(TransformerEmbeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = self.generate_positional_encodings(max_len, d_model).to(device)  # <-- Move to device
    
    
    def generate_positional_encodings(self, max_len, d_model):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        positional_encoding = torch.zeros(max_len, d_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)
        return positional_encoding

    def forward(self, x):
        # Get embeddings
        embeddings = self.embedding(x)
        
        # Add positional encodings
        return embeddings + self.positional_encoding[:x.size(1), :, :].expand(-1, x.size(0), -1).permute(1, 0, 2)




class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)
    
    def split_into_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.split_into_heads(self.wq(query), batch_size)
        key = self.split_into_heads(self.wk(key), batch_size)
        value = self.split_into_heads(self.wv(value), batch_size)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.depth**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)

        return self.dense(output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(torch.nn.functional.relu(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_output))

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, len(word2index))

    def forward(self, x, enc_output, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc_out(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_len=512, device='cpu'):  # <-- Add device parameter
        super(Transformer, self).__init__()
        self.embedding = TransformerEmbeddings(vocab_size, d_model, max_len, device)  # <-- Pass the device
        # ... [rest of the class remains unchanged]
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(self.embedding(src), src_mask)
        return self.decoder(self.embedding(tgt), enc_output, tgt_mask)


df['incorrect_encoded'] = pad_sequences(df['incorrect_encoded'].tolist(), word2index['<pad>'])
df['correct_encoded'] = pad_sequences(df['correct_encoded'].tolist(), word2index['<pad>'])


# Hyperparameters
EPOCHS = 100
LEARNING_RATE = 0.001

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model instantiation
transformer_model = Transformer(len(word2index), d_model=512, num_heads=8, d_ff=2048, num_layers=6, max_len=max_seq_len, device=device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer_model.parameters(), lr=LEARNING_RATE)

# Move model to GPU if available
transformer_model = transformer_model.to(device)

# Training loop
losses = []

for epoch in range(EPOCHS):
    total_loss = 0
    for batch_src, batch_tgt in train_loader:
        
        batch_src, batch_tgt = batch_src.to(device), batch_tgt.to(device)
        
        optimizer.zero_grad()
        
        outputs = transformer_model(batch_src, batch_tgt)
        outputs = outputs.view(-1, outputs.shape[-1])
        targets = batch_tgt.view(-1)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

print("Training finished.")

