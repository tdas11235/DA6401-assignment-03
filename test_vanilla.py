import torch.nn.functional as F
import csv
import torch
from data import get_dataloader
from model import SeqModel
from tester import Evaluator
from vocab import SPECIAL_TOKENS
import random
import numpy as np


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


seed_everything(100)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, input_vocab, target_vocab = get_dataloader(
    "./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv", batch_size=256, shuffle=True, resample=True)
test_loader, _, _ = get_dataloader("./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv",
                                   batch_size=256, input_vocab=input_vocab, target_vocab=target_vocab, shuffle=False, resample=False)
run_name = (
    f"ED{64}_"
    f"HD{256}_"
    f"CT{'GRU'}_"
    f"NE{3}_"
    f"ND{3}_"
    f"DO{0.2580857199720224}_LR{0.000974085143377538}_"
    f"BW{3}"
)
config = {
    'input_vocab_size': len(input_vocab),
    'target_vocab_size': len(target_vocab),
    'target_vocab': target_vocab,
    'input_vocab': input_vocab,
    'embedding_dim': 64,
    'hidden_dim': 256,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'cell_type': 'GRU',
    'dropout': 0.2580857199720224,
    'pad_idx': SPECIAL_TOKENS['<pad>'],
    'lr': 0.000974085143377538,
    'device': device,
    'model_name': run_name
}
model = SeqModel(config, len(input_vocab), len(target_vocab))
model.load_state_dict(torch.load(
    f"./final_checkpoints/{run_name}.pt", map_location=config['device']))
model.to(config['device'])
evaluator = Evaluator(model, input_vocab, target_vocab, device=device)
evaluator.evaluate_accuracy_vanilla(test_loader, use_beam=True, beam_width=3)
