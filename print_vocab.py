import torch

# Načítanie slovníka z súboru
vocab = torch.load('custom_vocab.pth')

# Vypísanie obsahu slovníka
for token, idx in vocab.items():
    print(f"{token}: {idx}")
