import torch
from data import get_dataloader
from model import SeqModel
from trainer import Trainer
from vocab import SPECIAL_TOKENS
import random
import numpy as np
import time

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def compute_exact_match_accuracy(model, data_loader, target_vocab, device, beam_width=5, max_len=30):
    model.eval()
    eos_token = target_vocab['</s>']
    sos_token = target_vocab['<s>']
    pad_token = target_vocab['<pad>']
    total = 0
    correct = 0

    with torch.no_grad():
        for src_batch, _, tgt_out_batch in data_loader:
            src_batch = src_batch.to(device)
            batch_size = src_batch.size(0)

            pred_sequences = model.beam_search_decode_batch(src_batch, beam_width, max_len)

            for i in range(batch_size):
                pred_indices = pred_sequences[i]
                # Clean prediction: remove <s> and truncate at </s>
                if pred_indices and pred_indices[0] == sos_token:
                    pred_indices = pred_indices[1:]
                if eos_token in pred_indices:
                    pred_indices = pred_indices[:pred_indices.index(eos_token)]

                # Clean reference: remove padding and truncate at </s>
                tgt_indices = tgt_out_batch[i].tolist()
                if eos_token in tgt_indices:
                    tgt_indices = tgt_indices[:tgt_indices.index(eos_token)]
                tgt_indices = [idx for idx in tgt_indices if idx != pad_token]

                if pred_indices == tgt_indices:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Exact-match (sequence-level) accuracy: {accuracy:.4f}")
    return accuracy

def compute_exact_match_accuracy_greedy(model, data_loader, target_vocab, device, max_len=30):
    model.eval()
    eos_token = target_vocab['</s>']
    sos_token = target_vocab['<s>']
    pad_token = target_vocab['<pad>']
    total = 0
    correct = 0

    with torch.no_grad():
        for src_batch, _, tgt_out_batch in data_loader:
            src_batch = src_batch.to(device)
            batch_size = src_batch.size(0)

            # Use greedy decode for each input
            # pred_sequences = [model.greedy_decode(src_batch[i], max_len=max_len) for i in range(batch_size)]
            pred_sequences = model.greedy_decode_batch(src_batch, max_len=max_len)

            for i in range(batch_size):
                pred_indices = pred_sequences[i]
                # Clean prediction: remove <s> and truncate at </s>
                if pred_indices and pred_indices[0] == sos_token:
                    pred_indices = pred_indices[1:]
                if eos_token in pred_indices:
                    pred_indices = pred_indices[:pred_indices.index(eos_token)]

                # Clean reference: remove padding and truncate at </s>
                tgt_indices = tgt_out_batch[i].tolist()
                if eos_token in tgt_indices:
                    tgt_indices = tgt_indices[:tgt_indices.index(eos_token)]
                tgt_indices = [idx for idx in tgt_indices if idx != pad_token]

                if pred_indices == tgt_indices:
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Exact-match (sequence-level) accuracy: {accuracy:.4f}")
    return accuracy


def transliterate_word(model, word, src_vocab, tgt_vocab, device):
    model.eval()
    # Tokenize and convert to indices
    src_indices = [src_vocab.get(char, src_vocab['<unk>']) for char in word]
    src_tensor = torch.tensor(src_indices, dtype=torch.long).to(device)

    with torch.no_grad():
        pred_indices = model.beam_search_decode(src_tensor)
        pred_indices = pred_indices.tolist()

    # Remove <s> and truncate at </s>
    # if pred_indices[0] == tgt_vocab['<s>']:
    #     pred_indices = pred_indices[1:]
    # if tgt_vocab['</s>'] in pred_indices:
    #     pred_indices = pred_indices[:pred_indices.index(tgt_vocab['</s>'])]

    # Build reverse vocab
    idx2char = {idx: char for char, idx in tgt_vocab.items()}
    pred_chars = [idx2char[idx] for idx in pred_indices]

    return ''.join(pred_chars)

def transliterate_word_greedy(model, word, src_vocab, tgt_vocab, device):
    model.eval()
    # Tokenize and convert to indices
    src_indices = [src_vocab.get(char, src_vocab['<unk>']) for char in word]
    src_tensor = torch.tensor(src_indices, dtype=torch.long).to(device)

    with torch.no_grad():
        pred_indices = model.greedy_decode(src_tensor)
    
    # Remove start token <s> if present at beginning
    # if pred_indices[0] == tgt_vocab['<s>']:
    #     pred_indices = pred_indices[1:]
    # # Remove everything after end token </s> (including </s>)
    # if tgt_vocab['</s>'] in pred_indices:
    #     pred_indices = pred_indices[:pred_indices.index(tgt_vocab['</s>'])]

    # Build reverse vocab
    idx2char = {idx: char for char, idx in tgt_vocab.items()}
    pred_chars = [idx2char.get(idx, '') for idx in pred_indices]

    return ''.join(pred_chars)




def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    train_loader, input_vocab, target_vocab = get_dataloader("dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv", batch_size=256, shuffle=True, resample=True)
    val_loader, _, _ = get_dataloader("./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv", batch_size=256, input_vocab=input_vocab, target_vocab=target_vocab, shuffle=False, resample=False)
    test_loader, _, _ = get_dataloader("./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv", batch_size=256, input_vocab=input_vocab, target_vocab=target_vocab, shuffle=False, resample=False)

    config = {
        'input_vocab_size': len(input_vocab),
        'target_vocab_size': len(target_vocab),
        'target_vocab': target_vocab,
        'input_vocab': input_vocab,
        'embedding_dim': 256,
        'hidden_dim': 256,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'cell_type': 'GRU',
        'dropout': 0.3,
        'pad_idx': SPECIAL_TOKENS['<pad>'],
        'lr': 1e-3,
        'beam_size': 3,
        'device': device,
        'model_name': "trial"
    }

    model = SeqModel(config, len(input_vocab), len(target_vocab))
    # model.load_state_dict(torch.load("checkpoints/trial.pt", map_location=config['device']))
    # model.to(config['device'])


    trainer = Trainer(model, train_loader, val_loader, device, config)
    trainer.train(target_vocab=target_vocab, num_epochs=20)
    # model = SeqModel(config, len(input_vocab), len(target_vocab))
    model.load_state_dict(torch.load("checkpoints/trial.pt", map_location=config['device']))
    model.to(config['device'])
    compute_exact_match_accuracy(model, val_loader, target_vocab, device, beam_width=2)
    compute_exact_match_accuracy(model, test_loader, target_vocab, device, beam_width=2)
    # word = "vishayyon"
    # # word = "ankganit"
    # output = transliterate_word(model, word, input_vocab, target_vocab, device)
    # with open("output.txt", "w", encoding="utf-8") as f:
    #     f.write(f"Input: {word} → Output: {output}\n")

if __name__ == "__main__":
    # seed 82: Exact-match (sequence-level) accuracy: 0.3336, Exact-match (sequence-level) accuracy: 0.3334, lr= 1e-3
    # seed 100: Exact-match (sequence-level) accuracy: 0.3339, Exact-match (sequence-level) accuracy: 0.3483, lr= 1e-3
    # seed 62: Exact-match (sequence-level) accuracy: 0.3421, Exact-match (sequence-level) accuracy: 0.3325, lr= 1e-3
    # seed 72: Exact-match (sequence-level) accuracy: 0.3320, Exact-match (sequence-level) accuracy: 0.3423
    startt = time.time()
    seed_everything(seed=100)
    main()
    endt = time.time()
    print(endt-startt)
