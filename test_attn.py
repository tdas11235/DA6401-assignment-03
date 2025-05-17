import torch
from data import get_dataloader
from attention_inference import SeqModel
from tester import Evaluator
from vocab import SPECIAL_TOKENS
import random
import numpy as np
import time
import wandb

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def evaluate():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    train_loader, input_vocab, target_vocab = get_dataloader("dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv", batch_size=256, shuffle=True, resample=False)
    test_loader, _, _ = get_dataloader("./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv", batch_size=256, input_vocab=input_vocab, target_vocab=target_vocab, shuffle=False, resample=False)
    run_name = (
        f"ED{256}_"
        f"HD{256}_"
        f"CT{'LSTM'}_"
        f"NE{2}_"
        f"ND{2}_"
        f"DO{0.25}_LR{0.001}_"
        f"BW{3}"
    )
    # wandb.run.name = run_name
    config = {
        'input_vocab_size': len(input_vocab),
        'target_vocab_size': len(target_vocab),
        'target_vocab': target_vocab,
        'input_vocab': input_vocab,
        'embedding_dim': 256,
        'hidden_dim': 256,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'cell_type': 'LSTM',
        'dropout': 0.25,
        'pad_idx': SPECIAL_TOKENS['<pad>'],
        'lr': 0.001,
        'device': device,
        'model_name': run_name
    }
    src_batch, _, tgt_batch = next(iter(test_loader))
    model = SeqModel(config, len(input_vocab), len(target_vocab))
    model.load_state_dict(torch.load(f"checkpoints_attn_nor/{run_name}.pt", map_location=config['device']))
    model.to(config['device'])
    evaluator = Evaluator(model, input_vocab, target_vocab, device)
    acc = evaluator.evaluate_accuracy_attention(test_loader, use_beam=True, beam_width=3)
    # wandb.log({"test_accuracy": acc})
    # evaluator.log_predictions(test_loader, num_samples=20, beam_width=3, greedy=False, attn=True)
    # evaluator.save_predictions_to_csv(test_loader, filename="predictions_attention.csv", num_samples=len(test_loader.dataset), beam_width=3, greedy=True, attn=True)
    # evaluator.plot_attention_heatmaps(src_batch, beam_width=3, greedy=False)

def main():
    wandb.init(project="da6401-a3-a-test-1")
    evaluate()

if __name__ == "__main__":
    seed_everything(100)
    main()
