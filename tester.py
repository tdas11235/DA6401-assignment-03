import torch
import torch.nn.functional as F
import wandb
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pickle
import random


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
        self.model.eval()
        src_tensor = src_tensor.unsqueeze(0).to(self.device)  # (1, seq_len)
        encoder_outputs, hidden = self.model.encoder(src_tensor)
        input_token = torch.tensor(
            [[self.tgt_vocab['<s>']]], device=self.device)
        decoded_indices = []
        attentions = []
        for _ in range(max_len):
            logits, hidden, attn_weights = self.model.decoder(
                input_token, hidden, encoder_outputs)
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
        sequences = [([self.tgt_vocab['<s>']], 0.0, hidden, [])]
        completed = []
        for _ in range(max_len):
            all_candidates = []
            for seq, score, hidden_state, attns in sequences:
                if seq[-1] == self.tgt_vocab['</s>']:
                    completed.append((seq, score, attns))
                    continue
                input_token = torch.tensor([[seq[-1]]], device=self.device)
                logits, new_hidden, attn_weights = self.model.decoder(
                    input_token, hidden_state, encoder_outputs)
                log_probs = F.log_softmax(logits, dim=1).squeeze(0)
                topk_log_probs, topk_indices = torch.topk(
                    log_probs, beam_width)
                for k in range(beam_width):
                    token = topk_indices[k].item()
                    token_score = topk_log_probs[k].item()
                    new_seq = seq + [token]
                    new_score = score + token_score
                    new_attns = attns + [attn_weights.squeeze(0).cpu()]
                    if isinstance(new_hidden, tuple):
                        hidden_clone = (
                            new_hidden[0].clone(), new_hidden[1].clone())
                    else:
                        hidden_clone = new_hidden.clone()
                    all_candidates.append(
                        (new_seq, new_score, hidden_clone, new_attns))
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[
                :beam_width]
            if len(completed) >= beam_width:
                break
        completed += [(seq, score, attns) for seq, score, _,
                      attns in sequences if seq[-1] == self.tgt_vocab['</s>']]
        completed = sorted(completed, key=lambda x: x[1], reverse=True)
        if len(completed) == 0:
            # Fall back to top beam candidate if none completed
            top_seq, top_score, _, top_attns = sequences[0]
            completed = [(top_seq, top_score, top_attns)]
        else:
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
        input_token = torch.tensor(
            [[self.tgt_vocab['<s>']]], device=self.device)
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
                logits, new_hidden = self.model.decoder(
                    input_token, hidden_state)
                log_probs = F.log_softmax(logits, dim=1).squeeze(0)
                topk_log_probs, topk_indices = torch.topk(
                    log_probs, beam_width)
                for k in range(beam_width):
                    token = topk_indices[k].item()
                    token_score = topk_log_probs[k].item()
                    new_seq = seq + [token]
                    new_score = score + token_score
                    if isinstance(new_hidden, tuple):
                        hidden_clone = (
                            new_hidden[0].clone(), new_hidden[1].clone())
                    else:
                        hidden_clone = new_hidden.clone()
                    all_candidates.append((new_seq, new_score, hidden_clone))
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[
                :beam_width]
            if len(completed) >= beam_width:
                break
        completed += [b for b in sequences if b[0]
                      [-1] == self.tgt_vocab['</s>']]
        completed = sorted(completed, key=lambda x: x[1], reverse=True)
        best_seq = completed[0][0]
        best_seq = best_seq[1:]  # remove <s>
        if self.tgt_vocab['</s>'] in best_seq:
            best_seq = best_seq[:best_seq.index(self.tgt_vocab['</s>'])]
        return best_seq

    def evaluate_accuracy_vanilla(self, data_loader, use_beam=False, beam_width=5, csv_path="transliteration_results_van.csv"):
        self.model.eval()
        correct = 0
        corrects = 0
        total = 0
        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['Input (Roman)', 'Predicted (Devanagari)', 'True (Devanagari)', 'Match'])

            for batch in data_loader:
                src_batch, _, tgt_batch = batch
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)

                for src_tensor, tgt_tensor in zip(src_batch, tgt_batch):
                    # --- Decode ground truth ---
                    tgt_tokens = tgt_tensor.tolist()
                    tgt_tokens = [t for t in tgt_tokens if t !=
                                  self.tgt_vocab['<pad>']]
                    if self.tgt_vocab['<s>'] in tgt_tokens:
                        tgt_tokens = tgt_tokens[1:]
                    if self.tgt_vocab['</s>'] in tgt_tokens:
                        tgt_tokens = tgt_tokens[:tgt_tokens.index(
                            self.tgt_vocab['</s>'])]
                    true_str = ''.join([self.idx2tgt.get(t, '')
                                       for t in tgt_tokens])
                    # --- Decode prediction ---
                    if use_beam:
                        pred_tokens = self.beam_search_decode_vanilla(
                            src_tensor, beam_width=beam_width)
                    else:
                        pred_tokens = self.greedy_decode_vanilla(src_tensor)

                    pred_str = ''.join([self.idx2tgt.get(t, '')
                                       for t in pred_tokens])
                    # --- Decode input (Romanized) ---
                    src_tokens = src_tensor.tolist()
                    src_tokens = [t for t in src_tokens if t !=
                                  self.src_vocab['<pad>']]
                    input_str = ''.join([self.idx2src.get(t, '')
                                        for t in src_tokens])
                    # --- Compare both token-wise and string-wise ---
                    if pred_tokens == tgt_tokens:
                        correct += 1
                    if pred_str == true_str:
                        corrects += 1
                    total += 1
                    # Write row to CSV
                    writer.writerow(
                        [input_str, pred_str, true_str, pred_str == true_str])

        accuracy = correct / total
        print(f"Token Accuracy: {accuracy * 100:.2f}%")
        print(f"String Accuracy: {corrects / total:.4f} ({corrects}/{total})")
        print(f"CSV results saved to: {csv_path}")
        return accuracy

    def evaluate_accuracy_attention(self, data_loader, use_beam=False, beam_width=5,
                                    csv_path="transliteration_results.csv",
                                    attention_dump_path="correct_attention_weights.pkl"):
        self.model.eval()
        correct = 0
        corrects = 0
        total = 0
        saved_attentions = []
        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['Input (Roman)', 'Predicted (Devanagari)', 'True (Devanagari)', 'Match'])
            for batch in data_loader:
                src_batch, _, tgt_batch = batch
                src_batch = src_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)
                for src_tensor, tgt_tensor in zip(src_batch, tgt_batch):
                    # Prepare ground truth
                    tgt_tokens = tgt_tensor.tolist()
                    tgt_tokens = [t for t in tgt_tokens if t !=
                                  self.tgt_vocab['<pad>']]
                    if self.tgt_vocab['<s>'] in tgt_tokens:
                        tgt_tokens = tgt_tokens[1:]
                    if self.tgt_vocab['</s>'] in tgt_tokens:
                        tgt_tokens = tgt_tokens[:tgt_tokens.index(
                            self.tgt_vocab['</s>'])]

                    true_str = ''.join([self.idx2tgt.get(t, '')
                                       for t in tgt_tokens])
                    # Decode prediction and get attention
                    if use_beam:
                        pred_tokens, attn_weights = self.beam_search_decode_attention(
                            src_tensor, beam_width=beam_width)
                    else:
                        pred_tokens, attn_weights = self.greedy_decode_attention(
                            src_tensor)

                    pred_str = ''.join([self.idx2tgt.get(t, '')
                                       for t in pred_tokens])
                    # Prepare input string
                    src_tokens = src_tensor.tolist()
                    src_tokens = [t for t in src_tokens if t !=
                                  self.src_vocab['<pad>']]
                    input_str = ''.join([self.idx2src.get(t, '')
                                        for t in src_tokens])
                    if pred_tokens == tgt_tokens:
                        correct += 1
                    if pred_str == true_str:
                        corrects += 1
                        saved_attentions.append({
                            "input_str": input_str,
                            "pred_str": pred_str,
                            "true_str": true_str,
                            "attn_weights": attn_weights.detach().cpu().numpy() if hasattr(attn_weights, "cpu") else attn_weights
                        })
                    total += 1
                    writer.writerow(
                        [input_str, pred_str, true_str, pred_str == true_str])
        accuracy = correct / total
        print(f"Token Accuracy: {accuracy * 100:.2f}%")
        print(f"String Accuracy: {corrects / total:.4f} ({corrects}/{total})")
        print(f"CSV results saved to: {csv_path}")
        with open(attention_dump_path, "wb") as f:
            pickle.dump(saved_attentions, f)
        print(f"Saved attention weights to {attention_dump_path}")
        return accuracy

    def plot_attention_heatmaps(self, file_path, max_samples=9):
        """Plot and log attention heatmaps for up to max_samples inputs in a 3x3 grid."""
        data = pickle.load(open('correct_attention_weights.pkl', 'rb'))
        n = max_samples
        data = random.sample(data, 3 * n)
        random.shuffle(data)
        fig, axes = plt.subplots(3, 3, figsize=(
            12, 12), constrained_layout=True)
        axes = axes.flatten()
        for i in range(n):
            pred_str = data[i]['pred_str']
            src_str = data[i]['input_str']
            pred_chars = list(pred_str)
            pred_chars.append("</s>")
            src_chars = list(src_str)
            attn = data[i]['attn_weights']
            attn = attn[:len(pred_chars), :len(src_chars)]
            im = axes[i].imshow(attn, cmap='viridis', aspect='auto')
            axes[i].set_title(f"Input: {src_str}")
            axes[i].set_xlabel("Source Tokens")
            axes[i].set_ylabel("Target Tokens")
            axes[i].set_yticks(range(len(pred_chars)))
            axes[i].set_yticklabels(pred_chars, fontproperties=prop)
            axes[i].set_xticks(range(len(src_chars)))
            axes[i].set_xticklabels(src_chars)
        for j in range(n, 9):
            axes[j].axis('off')
        # plt.tight_layout()
        cbar = fig.colorbar(
            im, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
        cbar.ax.tick_params(labelsize=8)
        plt.savefig("attn_maps.png", bbox_inches='tight', dpi=300)
        plt.show()
