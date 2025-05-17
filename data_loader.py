import pandas as pd
from vocab import build_vocab, SPECIAL_TOKENS
import random


def load_data(path):
    df = pd.read_csv(path, sep='\t', names=['target', 'input', 'attest'])
    df = df.dropna()
    df = df[df['input'].str.strip() != '</s> </s']
    return df['input'].tolist(), df['target'].tolist(), df['attest'].astype(int).tolist()

def encode_sequence(seq, vocab, add_start=False, add_end=False):
    tokens = []
    if add_start:
        tokens.append(vocab['<s>'])
    tokens += [vocab.get(ch, vocab['<unk>']) for ch in seq]
    if add_end:
        tokens.append(vocab['</s>'])
    return tokens

def decode_sequence(indices, inv_vocab):
    return "".join([inv_vocab.get(idx, "") for idx in indices])

def prepare_data(path, input_vocab=None, target_vocab=None):
    inputs, targets, attests = load_data(path)
    if input_vocab is None:
        input_vocab = build_vocab(inputs)
    if target_vocab is None:
        target_vocab = build_vocab(targets)
    encoded = []
    for inp, tgt, att in zip(inputs, targets, attests):
        src = encode_sequence(inp, input_vocab)
        tgt_in = encode_sequence(tgt, target_vocab, add_start=True)
        tgt_out = encode_sequence(tgt, target_vocab, add_end=True)
        encoded.append((src, tgt_in, tgt_out, att))
    return encoded, input_vocab, target_vocab