SPECIAL_TOKENS = {'<unk>': 0, '<s>': 1, '</s>': 2, '<pad>': 3}

def build_vocab(data, is_input=True):
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    idx = len(vocab)
    for word in data:
        chars = list(word.strip())
        for ch in chars:
            if ch not in vocab:
                vocab[ch] = idx
                idx += 1
    return vocab