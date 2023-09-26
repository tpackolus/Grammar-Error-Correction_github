import pandas as pd
from torchtext.data import get_tokenizer
import torch

def build_vocab_from_data(data_file, output_file):
    # 1. Načítajte dáta do DataFrame
    df = pd.read_csv(data_file, sep='\t')
    
    # 2. Tokenizujte vety z obidvoch stĺpcov
    tokenizer = get_tokenizer("basic_english")
    unique_tokens = set()
    
    for _, row in df.iterrows():
        tokens_incorrect = tokenizer(row['incorrect'])
        tokens_correct = tokenizer(row['correct'])
        unique_tokens.update(tokens_incorrect)
        unique_tokens.update(tokens_correct)
    
    # 3. Vytvorte slovník
    vocab = {'<UNK>': 0, '<PAD>': 1, '<BOS>': 2, '<EOS>': 3}
    for token in unique_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    
    # 4. Uložte slovník
    formatted_vocab = {k: f"{k} -> {v}" for k, v in vocab.items()}
    torch.save(formatted_vocab, output_file)

# Použite funkciu na vytvorenie slovníka
build_vocab_from_data("/mnt/sharedhome/harahus/Tvorenie_gram_chyb/skrec_low_modifed.csv", 'custom_vocab.pth')
