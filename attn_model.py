import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, num_layers, dropout, rnn_cell_type, device):
        super().__init__()
        cell = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_cell_type]
        self.device = device
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.cell = cell(embedding_dim, hidden_dim, num_layers=num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
    
    def forward(self, src):
        embedded = self.embedding(src)  # (batch, seq_len, emb_dim)
        outputs, hidden = self.cell(embedded)  # outputs: (batch, seq_len, hidden_dim)
        # outputs from last layer for all time steps returned (for attention)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand_as(encoder_outputs)
        e = torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden_expanded))
        scores = self.v(e).squeeze(2)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_dim, hidden_dim, num_layers, dropout, rnn_cell_type, device):
        super().__init__()
        cell = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_cell_type]
        self.device = device
        self.embedding = nn.Embedding(target_vocab_size, embedding_dim)
        self.cell = cell(embedding_dim + hidden_dim, hidden_dim, num_layers=num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim * 2, target_vocab_size)

    def forward(self, input_token, hidden, encoder_outputs):
        # input_token: (batch, 1)
        embedded = self.embedding(input_token)  # (batch, 1, emb_dim)
        if isinstance(hidden, tuple):  # LSTM
            dec_hidden = hidden[0][-1]  # (batch, hidden_dim)
        else:
            dec_hidden = hidden[-1]     # (batch, hidden_dim)
        context, attn_weights = self.attention(dec_hidden, encoder_outputs)  # (batch, hidden_dim)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)  # (batch, 1, emb+hidden)
        output, hidden = self.cell(rnn_input, hidden)  # output: (batch, 1, hidden_dim)
        output = output.squeeze(1)  # (batch, hidden_dim)
        concat_output = torch.cat([output, context], dim=1)  # (batch, hidden_dim*2)
        logits = self.output_layer(concat_output)  # (batch, target_vocab_size)
        return logits, hidden, attn_weights


class SeqModel(nn.Module):
    def __init__(self, config, input_vocab_size, target_vocab_size):
        super().__init__()
        self.device = config["device"]
        self.tgt_vocab = config["target_vocab"]
        self.sos_token = self.tgt_vocab['<s>']
        self.eos_token = self.tgt_vocab['</s>']

        self.encoder = Encoder(
            input_vocab_size=input_vocab_size,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_encoder_layers"],
            dropout=config["dropout"],
            rnn_cell_type=config["cell_type"],
            device=self.device,
        )
        self.decoder = Decoder(
            target_vocab_size=target_vocab_size,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_decoder_layers"],
            dropout=config["dropout"],
            rnn_cell_type=config["cell_type"],
            device=self.device,
        )

    def forward(self, src, tgt, teacher_forcing_ratio=0.0):
        batch_size = src.size(0)
        max_len = tgt.size(1)
        vocab_size = self.decoder.output_layer.out_features

        encoder_outputs, hidden = self.encoder(src)
        input_token = tgt[:, 0].unsqueeze(1)  # <s>
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(self.device)
        for t in range(max_len):
            logits, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t, :] = logits
            top1 = logits.argmax(1).unsqueeze(1)
            input_token = tgt[:, t].unsqueeze(1) if torch.rand(1).item() < teacher_forcing_ratio else top1
        return outputs

    def greedy_decode(self, src_seq, max_len=30):
        self.eval()
        with torch.no_grad():
            src_seq = src_seq.unsqueeze(0).to(self.device)
            encoder_outputs, hidden = self.encoder(src_seq)
            input_token = torch.tensor([[self.sos_token]], device=self.device)
            decoded_tokens = [self.sos_token]
            for _ in range(max_len):
                logits, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
                next_token = logits.argmax(dim=1).item()
                decoded_tokens.append(next_token)
                if next_token == self.eos_token:
                    break
                input_token = torch.tensor([[next_token]], device=self.device)

            return decoded_tokens

    def beam_search_decode(self, src_seq, beam_width=5, max_len=30):
        self.eval()
        with torch.no_grad():
            src_seq = src_seq.unsqueeze(0).to(self.device)
            encoder_outputs, hidden = self.encoder(src_seq)
            sequences = [([self.sos_token], 0.0, hidden)]
            completed = []
            for _ in range(max_len):
                all_candidates = []
                for seq, score, hidden_state in sequences:
                    if seq[-1] == self.eos_token:
                        completed.append((seq, score))
                        continue
                    input_token = torch.tensor([[seq[-1]]], device=self.device)
                    logits, new_hidden, _ = self.decoder(input_token, hidden_state, encoder_outputs)
                    log_probs = F.log_softmax(logits, dim=1).squeeze(0)
                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                    for i in range(beam_width):
                        token = topk_indices[i].item()
                        token_score = topk_log_probs[i].item()
                        new_seq = seq + [token]
                        new_score = score + token_score
                        # Clone hidden
                        if isinstance(new_hidden, tuple):
                            hidden_clone = (new_hidden[0].clone(), new_hidden[1].clone())
                        else:
                            hidden_clone = new_hidden.clone()
                        all_candidates.append((new_seq, new_score, hidden_clone))
                if not all_candidates:
                    break
                sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                if len(completed) >= beam_width:
                    break
            # If no sequences ended in <eos>, use the best current beam
            if not completed:
                completed = sequences
            completed = sorted(completed, key=lambda x: x[1], reverse=True)
            return completed[0][0]


    def greedy_decode_batch(self, src_batch, max_len=30):
        self.eval()
        batch_size = src_batch.size(0)
        src_batch = src_batch.to(self.device)

        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src_batch)
            input_token = torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=self.device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            decoded_tokens = torch.full((batch_size, max_len + 1), self.tgt_vocab['<pad>'], dtype=torch.long, device=self.device)
            decoded_tokens[:, 0] = self.sos_token

            for t in range(1, max_len + 1):
                logits, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
                next_token = logits.argmax(dim=1)
                decoded_tokens[:, t] = next_token
                finished = finished | (next_token == self.eos_token)
                if finished.all():
                    break
                input_token = next_token.unsqueeze(1)

        return decoded_tokens[:, 1:].tolist()
    
    def beam_search_decode_batch(self, src_batch, beam_width=5, max_len=30):
        self.eval()
        batch_size = src_batch.size(0)
        src_batch = src_batch.to(self.device)
        all_decoded = []
        with torch.no_grad():
            for i in range(batch_size):
                src_seq = src_batch[i].unsqueeze(0)  # (1, seq_len)
                encoder_outputs, hidden = self.encoder(src_seq)
                sequences = [([self.sos_token], 0.0, hidden)]
                completed = []
                for _ in range(max_len):
                    all_candidates = []
                    for seq, score, hidden_state in sequences:
                        if seq[-1] == self.eos_token:
                            completed.append((seq, score))
                            continue
                        input_token = torch.tensor([[seq[-1]]], device=self.device)
                        logits, new_hidden, _ = self.decoder(input_token, hidden_state, encoder_outputs)
                        log_probs = F.log_softmax(logits, dim=1).squeeze(0)
                        topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                        for k in range(beam_width):
                            token = topk_indices[k].item()
                            token_score = topk_log_probs[k].item()
                            new_seq = seq + [token]
                            new_score = score + token_score
                            if isinstance(new_hidden, tuple):  # LSTM
                                hidden_clone = (new_hidden[0].clone(), new_hidden[1].clone())
                            else:
                                hidden_clone = new_hidden.clone()

                            all_candidates.append((new_seq, new_score, hidden_clone))
                    sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
                    if len(completed) >= beam_width:
                        break
                completed += [b for b in sequences if b[0][-1] == self.eos_token]
                completed = sorted(completed, key=lambda x: x[1], reverse=True)
                best_seq = completed[0][0]
                all_decoded.append(best_seq)
        return all_decoded


