import torch

# Načítanie slovníka z .pth súboru
vocab_path = "/mnt/sharedhome/harahus/Grammar-Error-Correction_github/vocab/vocab_10K.pth"
vocab = torch.load(vocab_path)

# Výpis prvých 10 slov zo slovníka a ich indexov
for i, word in enumerate(list(vocab.get_itos())):
    if i >= 10:  # vypíše len prvých 10 slov
        break
    print(word, "->", vocab[word])