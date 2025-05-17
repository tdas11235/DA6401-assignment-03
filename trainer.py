import torch
import torch.nn as nn
import torch.optim as optim
import os
import wandb

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config, save_dir="checkpoints"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.criterion = nn.CrossEntropyLoss(ignore_index=config['pad_idx'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.model_name = config["model_name"]
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for src, tgt_in, tgt_out in self.train_loader:
            src, tgt_in, tgt_out = src.to(self.device), tgt_in.to(self.device), tgt_out.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(src, tgt_in)  # output: (batch, tgt_seq_len, vocab_size)
            # reshape for loss: (batch * seq_len, vocab_size)
            output = output.view(-1, output.size(-1))
            tgt_out = tgt_out.view(-1)

            loss = self.criterion(output, tgt_out)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.train_loader)
    
    def compute_accuracy_beam(self, data_loader, target_vocab, beam_width=5, max_len=30):
        self.model.eval()
        eos_token = target_vocab['</s>']
        sos_token = target_vocab['<s>']
        pad_token = target_vocab['<pad>']
        total = 0
        correct = 0
        with torch.no_grad():
            for src_batch, _, tgt_out_batch in data_loader:
                src_batch = src_batch.to(self.device)
                batch_size = src_batch.size(0)
                pred_sequences = self.model.beam_search_decode_batch(src_batch, beam_width, max_len)
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
        return accuracy
    
    def compute_accuracy_greedy(self, data_loader, target_vocab, max_len=30):
        self.model.eval()
        eos_token = target_vocab['</s>']
        sos_token = target_vocab['<s>']
        pad_token = target_vocab['<pad>']
        total = 0
        correct = 0
        with torch.no_grad():
            for src_batch, _, tgt_out_batch in data_loader:
                src_batch = src_batch.to(self.device)
                batch_size = src_batch.size(0)
                pred_sequences = self.model.greedy_decode_batch(src_batch, max_len=max_len)
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
        return accuracy

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for src, tgt_in, tgt_out in self.val_loader:
                src, tgt_in, tgt_out = src.to(self.device), tgt_in.to(self.device), tgt_out.to(self.device)
                output = self.model(src, tgt_in, teacher_forcing_ratio=0.0)
                output = output.view(-1, output.size(-1))
                tgt_out = tgt_out.view(-1)
                loss = self.criterion(output, tgt_out)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def save_checkpoint(self):
        checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)

    def train(self, target_vocab, num_epochs):
        best_val_acc = -float('inf')
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()
            train_acc = self.compute_accuracy_greedy(self.train_loader, target_vocab)
            val_acc = self.compute_accuracy_greedy(self.val_loader, target_vocab)
            print(f"Epoch {epoch}:\n Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint()
                print(f"Checkpoint saved at epoch {epoch}")
        return best_val_acc
