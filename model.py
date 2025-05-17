import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, num_layers, dropout, rnn_cell_type, device):
        super().__init__()
        cell = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_cell_type]
        self.device = device
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.cell = cell(embedding_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)

    def forward(self, src):
        embedded = self.embedding(src)  # (batch, seq_len, emb_dim)
        outputs, hidden = self.cell(embedded)  # outputs: all hidden states, hidden: final hidden state(s)
        return outputs, hidden
    

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_dim, hidden_dim, num_layers, dropout, rnn_cell_type, device):
        super().__init__()
        cell = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_cell_type]
        self.device = device
        self.embedding = nn.Embedding(target_vocab_size, embedding_dim)
        self.cell = cell(embedding_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, target_vocab_size)

    def forward(self, input_token, hidden):
        # input_token: (batch, 1)
        embedded = self.embedding(input_token)  # (batch, 1, emb_dim)
        output, hidden = self.cell(embedded, hidden)  # (batch, 1, hidden_dim)
        logits = self.output_layer(output.squeeze(1))  # (batch, vocab_size)
        return logits, hidden


class SeqModel(nn.Module):
    def __init__(self, config, input_vocab_size, target_vocab_size):
        super().__init__()
        self.device = config["device"]
        self.tgt_vocab = config["target_vocab"]

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
        input_token = tgt[:, 0].unsqueeze(1)  # first token to decoder (<s>)
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(self.device)
        for t in range(0, max_len):
            logits, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = logits
            top1 = logits.argmax(1).unsqueeze(1)
            input_token = tgt[:, t].unsqueeze(1) if torch.rand(1).item() < teacher_forcing_ratio else top1
        return outputs

    def beam_search_decode(self, src_seq, beam_width=5, max_len=30):
        sos_token = self.tgt_vocab['<s>']
        eos_token = self.tgt_vocab['</s>']
        src_seq = src_seq.unsqueeze(0)  # (1, seq_len)
        encoder_outputs, hidden = self.encoder(src_seq)
        beams = [(torch.tensor([sos_token], device=self.device), 0.0, hidden)]
        completed = []
        for _ in range(max_len):
            new_beams = []
            for seq, score, hidden in beams:
                if seq[-1].item() == eos_token:
                    completed.append((seq, score))
                    continue
                input_token = seq[-1].view(1, 1)  # (1,1)
                logits, hidden_new = self.decoder(input_token, hidden)
                log_probs = torch.log_softmax(logits, dim=-1)
                topk_log_probs, topk_indices = log_probs.topk(beam_width)
                for i in range(beam_width):
                    next_seq = torch.cat([seq, topk_indices[0, i].view(1)])
                    next_score = score + topk_log_probs[0, i].item()
                    # Clone hidden properly
                    if isinstance(hidden_new, tuple):
                        hidden_clone = (hidden_new[0].clone(), hidden_new[1].clone())
                    else:
                        hidden_clone = hidden_new.clone()
                    new_beams.append((next_seq, next_score, hidden_clone))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if len(completed) >= beam_width:
                break
        completed += [b for b in beams if b[0][-1].item() == eos_token]
        completed = sorted(completed, key=lambda x: x[1], reverse=True)
        return completed[0][0]

    
    def beam_search_decode_batch(self, src_batch, beam_width=5, max_len=30):
        src_batch = src_batch.to(self.device)
        batch_size = src_batch.size(0)
        decoded = []
        for i in range(batch_size):
            output = self.beam_search_decode(src_batch[i], beam_width, max_len)
            decoded.append(output.tolist())
        return decoded
    
    def greedy_decode(self, src_seq, max_len=30):
        sos_token = self.tgt_vocab['<s>']
        eos_token = self.tgt_vocab['</s>']

        self.eval()
        src_seq = src_seq.unsqueeze(0).to(self.device)  # (1, seq_len)

        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src_seq)  # Get encoder outputs and hidden state

            input_token = torch.tensor([[sos_token]], device=self.device)  # Start with <s>
            decoded_tokens = [sos_token]

            for _ in range(max_len):
                logits, hidden = self.decoder(input_token, hidden)  # logits: (1, 1, vocab_size)
                next_token = torch.argmax(logits.squeeze(1), dim=-1)  # (1,)
                next_token_id = next_token.item()
                decoded_tokens.append(next_token_id)

                if next_token_id == eos_token:
                    break

                input_token = next_token.unsqueeze(0)  # Make it (1, 1) again

        return decoded_tokens
    
    def greedy_decode_batch(self, src_batch, max_len=30):
        self.eval()
        sos_token = self.tgt_vocab['<s>']
        eos_token = self.tgt_vocab['</s>']
        batch_size = src_batch.size(0)
        src_batch = src_batch.to(self.device)

        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src_batch)  # hidden: (num_layers, batch, hidden_size)

            input_token = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=self.device)  # (B, 1)
            decoded_tokens = torch.full((batch_size, max_len + 1), self.tgt_vocab['<pad>'], dtype=torch.long, device=self.device)
            decoded_tokens[:, 0] = sos_token  # Set initial token

            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            for t in range(1, max_len + 1):
                logits, hidden = self.decoder(input_token, hidden)  # logits: (B, 1, vocab_size)
                next_token = torch.argmax(logits.squeeze(1), dim=-1)  # (B,)
                decoded_tokens[:, t] = next_token

                # Stop decoding for sequences that already produced <eos>
                finished |= next_token == eos_token
                if finished.all():
                    break

                input_token = next_token.unsqueeze(1)  # (B, 1)

        # Convert decoded tensor to list of lists, trimming at eos
        decoded_sequences = []
        for seq in decoded_tokens.tolist():
            if eos_token in seq:
                idx = seq.index(eos_token)
                decoded_sequences.append(seq[:idx + 1])
            else:
                decoded_sequences.append(seq)

        return decoded_sequences


