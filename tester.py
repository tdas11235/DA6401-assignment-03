import torch
import torch.nn.functional as F
import wandb
import csv
import matplotlib.pyplot as plt
from matplotlib.table import Table
import io
import numpy as np
from PIL import Image
import matplotlib.font_manager as fm


font_path = "./fonts/NotoSansDevanagari-Regular.ttf"
prop = fm.FontProperties(fname=font_path)

class Evaluator:
    def __init__(self, model, src_vocab, tgt_vocab, device):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device
        self.idx2src = {idx: char for char, idx in src_vocab.items()}
        self.idx2tgt = {idx: char for char, idx in tgt_vocab.items()}

    def greedy_decode_attention(self, src_tensor, max_len=30):
        """Greedy decode with attention weights extraction."""
        self.model.eval()
        src_tensor = src_tensor.unsqueeze(0).to(self.device)  # (1, seq_len)
        encoder_outputs, hidden = self.model.encoder(src_tensor)
        input_token = torch.tensor([[self.tgt_vocab['<s>']]], device=self.device)
        decoded_indices = []
        attentions = []
        for _ in range(max_len):
            logits, hidden, attn_weights = self.model.decoder(input_token, hidden, encoder_outputs)
            attentions.append(attn_weights.squeeze(0).cpu())  # (src_seq_len,)
            top1 = logits.argmax(1)
            token = top1.item()
            if token == self.tgt_vocab['</s>']:
                break
            decoded_indices.append(token)
            input_token = top1.unsqueeze(1)
        attentions = torch.stack(attentions)  # (tgt_len, src_len)
        return decoded_indices, attentions

    def beam_search_decode_attention(self, src_tensor, beam_width=5, max_len=30):
        self.model.eval()
        src_tensor = src_tensor.unsqueeze(0).to(self.device)
        encoder_outputs, hidden = self.model.encoder(src_tensor)
        # Each candidate: (sequence, score, hidden_state, attentions)
        sequences = [([self.tgt_vocab['<s>']], 0.0, hidden, [])]
        completed = []
        for _ in range(max_len):
            all_candidates = []
            for seq, score, hidden_state, attns in sequences:
                if seq[-1] == self.tgt_vocab['</s>']:
                    completed.append((seq, score, attns))
                    continue 
                input_token = torch.tensor([[seq[-1]]], device=self.device)
                logits, new_hidden, attn_weights = self.model.decoder(input_token, hidden_state, encoder_outputs)
                log_probs = F.log_softmax(logits, dim=1).squeeze(0)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width) 
                for k in range(beam_width):
                    token = topk_indices[k].item()
                    token_score = topk_log_probs[k].item()
                    new_seq = seq + [token]
                    new_score = score + token_score
                    new_attns = attns + [attn_weights.squeeze(0).cpu()]  # Append current step attention
                    if isinstance(new_hidden, tuple):
                        hidden_clone = (new_hidden[0].clone(), new_hidden[1].clone())
                    else:
                        hidden_clone = new_hidden.clone() 
                    all_candidates.append((new_seq, new_score, hidden_clone, new_attns))
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            if len(completed) >= beam_width:
                break
        completed += [(seq, score, attns) for seq, score, _, attns in sequences if seq[-1] == self.tgt_vocab['</s>']]
        completed = sorted(completed, key=lambda x: x[1], reverse=True)
        if len(completed) == 0:
            # Fall back to top beam candidate if none completed
            top_seq, top_score, _, top_attns = sequences[0]
            completed = [(top_seq, top_score, top_attns)]
        else:
            # Otherwise ensure only (seq, score, attns) in completed
            pass
        best_seq, _, best_attns = completed[0]
        best_seq = best_seq[1:]  # remove <s>
        if self.tgt_vocab['</s>'] in best_seq:
            best_seq = best_seq[:best_seq.index(self.tgt_vocab['</s>'])]
        attentions = torch.stack(best_attns)  # (tgt_len, src_len)
        return best_seq, attentions

    def greedy_decode_vanilla(self, src_tensor, max_len=30):
        self.model.eval()
        src_tensor = src_tensor.unsqueeze(0).to(self.device)  # (1, seq_len)
        encoder_outputs, hidden = self.model.encoder(src_tensor)
        input_token = torch.tensor([[self.tgt_vocab['<s>']]], device=self.device)
        decoded_indices = []
        for _ in range(max_len):
            logits, hidden = self.model.decoder(input_token, hidden)
            top1 = logits.argmax(1)
            token = top1.item()
            if token == self.tgt_vocab['</s>']:
                break
            decoded_indices.append(token)
            input_token = top1.unsqueeze(1)
        return decoded_indices

    def beam_search_decode_vanilla(self, src_tensor, beam_width=5, max_len=30):
        self.model.eval()
        src_tensor = src_tensor.unsqueeze(0).to(self.device)
        encoder_outputs, hidden = self.model.encoder(src_tensor)
        sequences = [([self.tgt_vocab['<s>']], 0.0, hidden)]
        completed = []
        for _ in range(max_len):
            all_candidates = []
            for seq, score, hidden_state in sequences:
                if seq[-1] == self.tgt_vocab['</s>']:
                    completed.append((seq, score))
                    continue   
                input_token = torch.tensor([[seq[-1]]], device=self.device)
                logits, new_hidden = self.model.decoder(input_token, hidden_state)
                log_probs = F.log_softmax(logits, dim=1).squeeze(0)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                for k in range(beam_width):
                    token = topk_indices[k].item()
                    token_score = topk_log_probs[k].item()
                    new_seq = seq + [token]
                    new_score = score + token_score
                    
                    if isinstance(new_hidden, tuple):
                        hidden_clone = (new_hidden[0].clone(), new_hidden[1].clone())
                    else:
                        hidden_clone = new_hidden.clone()
                    all_candidates.append((new_seq, new_score, hidden_clone))
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            if len(completed) >= beam_width:
                break
        completed += [b for b in sequences if b[0][-1] == self.tgt_vocab['</s>']]
        completed = sorted(completed, key=lambda x: x[1], reverse=True)
        best_seq = completed[0][0]
        best_seq = best_seq[1:]  # remove <s>
        if self.tgt_vocab['</s>'] in best_seq:
            best_seq = best_seq[:best_seq.index(self.tgt_vocab['</s>'])]
        return best_seq
    
    # def evaluate_accuracy_vanilla(self, data_loader, use_beam=False, beam_width=5):
    #     """Evaluate transliteration accuracy on the given data loader."""
    #     self.model.eval()
    #     correct = 0
    #     total = 0
    #     for batch in data_loader:
    #         src_batch, _, tgt_batch = batch
    #         src_batch = src_batch.to(self.device)
    #         tgt_batch = tgt_batch.to(self.device)
    #         for src_tensor, tgt_tensor in zip(src_batch, tgt_batch):
    #             tgt_tokens = tgt_tensor.tolist()
    #             # Remove special tokens from ground truth
    #             if self.tgt_vocab['<s>'] in tgt_tokens:
    #                 tgt_tokens.remove(self.tgt_vocab['<s>'])
    #             if self.tgt_vocab['</s>'] in tgt_tokens:
    #                 tgt_tokens = tgt_tokens[:tgt_tokens.index(self.tgt_vocab['</s>'])]
    #             # Decode prediction
    #             if use_beam:
    #                 pred_tokens = self.beam_search_decode_vanilla(src_tensor, beam_width=beam_width)
    #             else:
    #                 pred_tokens = self.greedy_decode_vanilla(src_tensor)
    #             # Compare predicted and true sequences
    #             if pred_tokens == tgt_tokens:
    #                 correct += 1
    #             total += 1
    #     accuracy = correct / total * 100
    #     print(f"Test Accuracy: {accuracy:.2f}%")
    #     return accuracy

    def evaluate_accuracy_vanilla(self, data_loader, use_beam=False, beam_width=5, csv_path="transliteration_results_van.csv"):
        """Evaluate transliteration accuracy and save results to CSV."""
        self.model.eval()
        correct = 0
        corrects = 0
        total = 0
        # Open CSV file to store results
        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Input (Roman)', 'Predicted (Devanagari)', 'True (Devanagari)', 'Match'])

            for batch in data_loader:
                src_batch, _, tgt_batch = batch
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)

                for src_tensor, tgt_tensor in zip(src_batch, tgt_batch):
                    # --- Decode ground truth ---
                    tgt_tokens = tgt_tensor.tolist()
                    tgt_tokens = [t for t in tgt_tokens if t != self.tgt_vocab['<pad>']]
                    if self.tgt_vocab['<s>'] in tgt_tokens:
                        tgt_tokens = tgt_tokens[1:]
                    if self.tgt_vocab['</s>'] in tgt_tokens:
                        tgt_tokens = tgt_tokens[:tgt_tokens.index(self.tgt_vocab['</s>'])]

                    true_str = ''.join([self.idx2tgt.get(t, '') for t in tgt_tokens])

                    # --- Decode prediction ---
                    if use_beam:
                        pred_tokens = self.beam_search_decode_vanilla(src_tensor, beam_width=beam_width)
                    else:
                        pred_tokens = self.greedy_decode_vanilla(src_tensor)

                    pred_str = ''.join([self.idx2tgt.get(t, '') for t in pred_tokens])

                    # --- Decode input (Romanized) ---
                    src_tokens = src_tensor.tolist()
                    src_tokens = [t for t in src_tokens if t != self.src_vocab['<pad>']]
                    input_str = ''.join([self.idx2src.get(t, '') for t in src_tokens])

                    # --- Compare both token-wise and string-wise ---
                    if pred_tokens == tgt_tokens:
                        correct += 1
                    if pred_str == true_str:
                        corrects += 1

                    total += 1

                    # Write row to CSV
                    writer.writerow([input_str, pred_str, true_str, pred_str == true_str])

        accuracy = correct / total
        print(f"Token Accuracy: {accuracy * 100:.2f}%")
        print(f"String Accuracy: {corrects / total:.4f} ({corrects}/{total})")
        print(f"CSV results saved to: {csv_path}")
        return accuracy

    # def evaluate_accuracy_attention(self, data_loader, use_beam=False, beam_width=5):
    #     """Evaluate transliteration accuracy on the given data loader."""
    #     self.model.eval()
    #     correct = 0
    #     total = 0
    #     for batch in data_loader:
    #         src_batch, _, tgt_batch = batch
    #         src_batch = src_batch.to(self.device)
    #         tgt_batch = tgt_batch.to(self.device)
    #         for src_tensor, tgt_tensor in zip(src_batch, tgt_batch):
    #             tgt_tokens = tgt_tensor.tolist()
    #             # Remove special tokens from ground truth
    #             if self.tgt_vocab['<s>'] in tgt_tokens:
    #                 tgt_tokens.remove(self.tgt_vocab['<s>'])
    #             if self.tgt_vocab['</s>'] in tgt_tokens:
    #                 tgt_tokens = tgt_tokens[:tgt_tokens.index(self.tgt_vocab['</s>'])]
    #             # Decode prediction
    #             if use_beam:
    #                 pred_tokens, _ = self.beam_search_decode_attention(src_tensor, beam_width=beam_width)
    #             else:
    #                 pred_tokens, _ = self.greedy_decode_attention(src_tensor)
    #             # Compare predicted and true sequences
    #             if pred_tokens == tgt_tokens:
    #                 correct += 1
    #             total += 1
    #     accuracy = correct / total
    #     print(f"Test Accuracy: {accuracy * 100:.2f}%")
    #     return accuracy


    # def evaluate_accuracy_attention(self, data_loader, use_beam=False, beam_width=5):
    #     """Evaluate transliteration accuracy on the given data loader."""
    #     self.model.eval()
    #     correct = 0
    #     total = 0
    #     corrects = 0
    #     for batch in data_loader:
    #         src_batch, _, tgt_batch = batch
    #         src_batch = src_batch.to(self.device)
    #         tgt_batch = tgt_batch.to(self.device)

    #         for src_tensor, tgt_tensor in zip(src_batch, tgt_batch):
    #             tgt_tokens = tgt_tensor.tolist()
    #             # Remove padding and special tokens
    #             tgt_tokens = [t for t in tgt_tokens if t != self.tgt_vocab['<pad>']]
    #             if self.tgt_vocab['<s>'] in tgt_tokens:
    #                 tgt_tokens = tgt_tokens[1:]  # remove <s>
    #             if self.tgt_vocab['</s>'] in tgt_tokens:
    #                 tgt_tokens = tgt_tokens[:tgt_tokens.index(self.tgt_vocab['</s>'])]

    #             # Decode prediction
    #             if use_beam:
    #                 pred_tokens, _ = self.beam_search_decode_attention(src_tensor, beam_width=beam_width)
    #             else:
    #                 pred_tokens, _ = self.greedy_decode_attention(src_tensor)

    #             # Ensure both lists are comparable
    #             if pred_tokens == tgt_tokens:
    #                 correct += 1
    #             pred_str = ''.join([self.idx2tgt.get(t, '') for t in pred_tokens])
    #             true_str = ''.join([self.idx2tgt.get(t, '') for t in tgt_tokens])
    #             corrects += (pred_str == true_str)
    #             total += 1

    #     accuracy = correct / total
    #     print(f"Test Accuracy: {accuracy * 100:.2f}%")
    #     print(f"True accuracy: {corrects / total}")
    #     return accuracy
    def evaluate_accuracy_attention(self, data_loader, use_beam=False, beam_width=5, csv_path="transliteration_results.csv"):
        """Evaluate transliteration accuracy and save results to CSV."""
        self.model.eval()
        correct = 0
        corrects = 0
        total = 0
        # Open CSV file to store results
        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Input (Roman)', 'Predicted (Devanagari)', 'True (Devanagari)', 'Match'])

            for batch in data_loader:
                src_batch, _, tgt_batch = batch
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)

                for src_tensor, tgt_tensor in zip(src_batch, tgt_batch):
                    # --- Decode ground truth ---
                    tgt_tokens = tgt_tensor.tolist()
                    tgt_tokens = [t for t in tgt_tokens if t != self.tgt_vocab['<pad>']]
                    if self.tgt_vocab['<s>'] in tgt_tokens:
                        tgt_tokens = tgt_tokens[1:]
                    if self.tgt_vocab['</s>'] in tgt_tokens:
                        tgt_tokens = tgt_tokens[:tgt_tokens.index(self.tgt_vocab['</s>'])]

                    true_str = ''.join([self.idx2tgt.get(t, '') for t in tgt_tokens])

                    # --- Decode prediction ---
                    if use_beam:
                        pred_tokens, _ = self.beam_search_decode_attention(src_tensor, beam_width=beam_width)
                    else:
                        pred_tokens, _ = self.greedy_decode_attention(src_tensor)

                    pred_str = ''.join([self.idx2tgt.get(t, '') for t in pred_tokens])

                    # --- Decode input (Romanized) ---
                    src_tokens = src_tensor.tolist()
                    src_tokens = [t for t in src_tokens if t != self.src_vocab['<pad>']]
                    input_str = ''.join([self.idx2src.get(t, '') for t in src_tokens])

                    # --- Compare both token-wise and string-wise ---
                    if pred_tokens == tgt_tokens:
                        correct += 1
                    if pred_str == true_str:
                        corrects += 1

                    total += 1

                    # Write row to CSV
                    writer.writerow([input_str, pred_str, true_str, pred_str == true_str])

        accuracy = correct / total
        print(f"Token Accuracy: {accuracy * 100:.2f}%")
        print(f"String Accuracy: {corrects / total:.4f} ({corrects}/{total})")
        print(f"CSV results saved to: {csv_path}")
        return accuracy



    def transliterate_word_greedy(self, word, attn=True):
        src_indices = [self.src_vocab.get(c, self.src_vocab['<unk>']) for c in word]
        src_tensor = torch.tensor(src_indices, dtype=torch.long)
        if attn:
            pred_indices, _ = self.greedy_decode_attention(src_tensor)
        else:
            pred_indices = self.greedy_decode_vanilla(src_tensor)
        pred_chars = [self.idx2tgt.get(idx, '') for idx in pred_indices]
        return ''.join(pred_chars)
    
    def transliterate_word_beam_search(self, word, beam_width=5, attn=True):
        src_indices = [self.src_vocab.get(c, self.src_vocab['<unk>']) for c in word]
        src_tensor = torch.tensor(src_indices, dtype=torch.long)
        if attn:
            pred_indices, _ = self.beam_search_decode_attention(src_tensor, beam_width=beam_width)
        else:
            pred_indices = self.beam_search_decode_vanilla(src_tensor, beam_width=beam_width)
        pred_chars = [self.idx2tgt.get(idx, '') for idx in pred_indices]
        return ''.join(pred_chars)

    def log_predictions(self, dataloader, num_samples=10, beam_width=5, greedy=True, attn=True):
        self.model.eval()
        rows = []
        count = 0
        for src_batch, _, tgt_batch in dataloader:
            src_batch, tgt_batch = src_batch.to(self.device), tgt_batch.to(self.device)
            for i in range(src_batch.size(0)):
                input_str = ''.join([self.idx2src.get(idx.item(), '') for idx in src_batch[i] if idx.item() != self.src_vocab['<pad>']])
                tgt_indices = tgt_batch[i].tolist()
                if self.tgt_vocab['<s>'] in tgt_indices:
                    tgt_indices.remove(self.tgt_vocab['<s>'])
                if self.tgt_vocab['</s>'] in tgt_indices:
                    tgt_indices = tgt_indices[:tgt_indices.index(self.tgt_vocab['</s>'])]
                true_str = ''.join([self.idx2tgt.get(idx, '') for idx in tgt_indices if idx != self.tgt_vocab['<pad>']])
                if greedy:
                    pred_str = self.transliterate_word_greedy(input_str, attn=attn)
                else:
                    pred_str = self.transliterate_word_beam_search(input_str, beam_width=beam_width, attn=attn)
                correct = pred_str == true_str
                rows.append((input_str, pred_str, true_str, correct))
                count += 1
                if count >= num_samples:
                    break
            if count >= num_samples:
                break

        # Create the image table
        fig, ax = plt.subplots(figsize=(12, 0.5 + 0.5 * len(rows)))
        ax.axis('off')
        tbl = Table(ax, bbox=[0, 0, 1, 1])
        ncols = 3
        col_labels = ['Input', 'Predicted', 'True']
        col_widths = [1, 1, 1]
        for j, label in enumerate(col_labels):
            tbl.add_cell(0, j, col_widths[j], 0.5, text=label, loc='center', facecolor='lightgray')
        for i, (inp, pred, true, correct) in enumerate(rows):
            colors = ['white', 'lightgreen' if correct else 'lightcoral', 'lightyellow']
            values = [inp, pred, true]
            for j, val in enumerate(values):
                tbl.add_cell(i + 1, j, col_widths[j], 0.5, text=val, loc='center', facecolor=colors[j])
        fig.canvas.draw() 
        for (row, col), cell in tbl.get_celld().items():
            cell.set_fontsize(10)
            if row > 0 and col in [1, 2]:
                cell.get_text().set_fontproperties(prop)
        ax.add_table(tbl)
        # Save to buffer and log to wandb
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        wandb.log({"Prediction Table": wandb.Image(img)})
        plt.close()

    def save_predictions_to_csv(self, dataloader, filename="predictions.csv", num_samples=100, beam_width=5, greedy=True, attn=True):
        self.model.eval()
        rows = [("Input", "Predicted", "True", "Correct_or_Incorrect")]
        count = 0
        for src_batch, _, tgt_batch in dataloader:
            src_batch, tgt_batch = src_batch.to(self.device), tgt_batch.to(self.device)
            for i in range(src_batch.size(0)):
                input_str = ''.join([self.idx2src.get(idx.item(), '') for idx in src_batch[i] if idx.item() != self.src_vocab['<pad>']])
                tgt_indices = tgt_batch[i].tolist()
                if self.tgt_vocab['<s>'] in tgt_indices:
                    tgt_indices.remove(self.tgt_vocab['<s>'])
                if self.tgt_vocab['</s>'] in tgt_indices:
                    tgt_indices = tgt_indices[:tgt_indices.index(self.tgt_vocab['</s>'])]
                true_str = ''.join([self.idx2tgt.get(idx, '') for idx in tgt_indices if idx != self.tgt_vocab['<pad>']])
                if greedy:
                    pred_str = self.transliterate_word_greedy(input_str, attn=attn)
                else:
                    pred_str = self.transliterate_word_beam_search(input_str, beam_width=beam_width, attn=attn)
                correct = int(pred_str == true_str)
                rows.append((input_str, pred_str, true_str, correct))
                count += 1
                if count >= num_samples:
                    break
            if count >= num_samples:
                break
        with open(filename, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"Saved {count} predictions to {filename}")

    def plot_attention_heatmaps(self, src_batch, max_samples=9, beam_width=5, greedy=True):
        """Plot and log attention heatmaps for up to max_samples inputs in a 3x3 grid."""
        self.model.eval()
        src_batch = src_batch.to(self.device)
        n = min(max_samples, src_batch.size(0))
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        for i in range(n):
            src_seq = src_batch[i]
            src_chars = [self.idx2src.get(idx.item(), '') for idx in src_seq if idx.item() != self.src_vocab['<pad>']]
            src_str = ''.join(src_chars)
            if greedy:
                _, attn_weights = self.greedy_decode_attention(src_seq)
            else:
                _, attn_weights = self.beam_search_decode_attention(src_seq, beam_width=beam_width)
            attn = attn_weights.detach().numpy()  # (tgt_len, src_len)
            axes[i].imshow(attn, cmap='viridis')
            axes[i].set_title(f"Input: {src_str}")
            axes[i].set_xlabel("Source Tokens")
            axes[i].set_ylabel("Target Tokens")
            axes[i].set_xticks(range(len(src_chars)))
            axes[i].set_xticklabels(src_chars, rotation=90, fontproperties=prop)
            # Target tokens chars unknown here, could be just indices or blanks
            axes[i].set_yticks([])
        for j in range(n, 9):
            axes[j].axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        wandb.log({"Attention Heatmaps": wandb.Image(img)})
        plt.close()

