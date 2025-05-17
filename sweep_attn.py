import wandb
import yaml
import torch
import torch.nn as nn
import gc
from data import get_dataloader
from attn_model import SeqModel
from trainer import Trainer
from vocab import SPECIAL_TOKENS
import random
import numpy as np

SEED = 100
COUNT = 10
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
TEST_PATH = "./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv"
TRAIN_PATH = "dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
DEV_PATH = "./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(path="attn_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def compute_accuracy_beam(model, data_loader, target_vocab, device, beam_width=5, max_len=30):
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
    

def sweep_func():
    wandb.init()
    wconfig = wandb.config
    train_loader, input_vocab, target_vocab = get_dataloader(TRAIN_PATH, batch_size=256, shuffle=True, resample=False)
    val_loader, _, _ = get_dataloader(DEV_PATH, batch_size=256, input_vocab=input_vocab, target_vocab=target_vocab, shuffle=False, resample=False)
    run_name = (
        f"ED{wconfig.embedding_dim}_"
        f"HD{wconfig.hidden_dim}_"
        f"CT{wconfig.cell_type}_"
        f"NE{wconfig.num_layers}_"
        f"ND{wconfig.num_layers}_"
        f"DO{wconfig.dropout}_LR{wconfig.lr}_"
        f"BW{wconfig.beam_width}"
    )
    wandb.run.name = run_name
    config = {
        'input_vocab_size': len(input_vocab),
        'target_vocab_size': len(target_vocab),
        'target_vocab': target_vocab,
        'input_vocab': input_vocab,
        'embedding_dim': wconfig.embedding_dim,
        'hidden_dim': wconfig.hidden_dim,
        'num_encoder_layers': wconfig.num_layers,
        'num_decoder_layers': wconfig.num_layers,
        'cell_type': wconfig.cell_type,
        'dropout': wconfig.dropout,
        'pad_idx': SPECIAL_TOKENS['<pad>'],
        'lr': wconfig.lr,
        'device': DEVICE,
        'model_name': run_name
    }
    model = SeqModel(config, len(input_vocab), len(target_vocab))
    trainer = Trainer(model, train_loader, val_loader, DEVICE, config, save_dir="checkpoints_attn_nor_low_100")
    best_val_acc = trainer.train(target_vocab=target_vocab, num_epochs=20)
    if wconfig.beam_width > 1:
        model.load_state_dict(torch.load(f"checkpoints_attn_nor_low_100/{run_name}.pt", map_location=config['device']))
        model.to(config['device'])
        best_val_acc = compute_accuracy_beam(model, val_loader, target_vocab, DEVICE, beam_width=wconfig.beam_width)
    wandb.log({"best_val_acc": best_val_acc})
    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

def main():
    sweep_config = load_config()
    project = "da6401-a3-a-nor-low-100"
    sweep_id = wandb.sweep(sweep_config, project=project)
    wandb.agent(sweep_id, sweep_func, count=COUNT)

if __name__ == "__main__":
    seed_everything(seed=SEED)
    main()