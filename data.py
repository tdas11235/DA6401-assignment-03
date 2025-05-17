import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

from data_loader import prepare_data
from vocab import SPECIAL_TOKENS

class DakshinaDataset(Dataset):
    def __init__(self, encoded_data):
        self.data = encoded_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    pad_idx = SPECIAL_TOKENS['<pad>']
    src_seqs = [torch.tensor(x[0], dtype=torch.long) for x in batch]
    tgt_in_seqs = [torch.tensor(x[1], dtype=torch.long) for x in batch]
    tgt_out_seqs = [torch.tensor(x[2], dtype=torch.long) for x in batch]
    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=pad_idx)
    tgt_in_padded = pad_sequence(tgt_in_seqs, batch_first=True, padding_value=pad_idx)
    tgt_out_padded = pad_sequence(tgt_out_seqs, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_in_padded, tgt_out_padded


def get_dataloader(path, batch_size=32, input_vocab=None, target_vocab=None, shuffle=True, resample=True):
    encoded_data, input_vocab, target_vocab = prepare_data(path, input_vocab, target_vocab)
    dataset = DakshinaDataset(encoded_data)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    if resample:
        # Extract attestation column and apply smoothing
        alpha = 1.0
        attests = [x[3] for x in encoded_data]
        weights = [att ** alpha for att in attests]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return dataloader, input_vocab, target_vocab
