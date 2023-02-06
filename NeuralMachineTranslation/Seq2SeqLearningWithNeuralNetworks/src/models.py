import os
import sys
import torch
from torch import nn


class Seq2SeqNMT(nn.Module):
    def __init__(self, config):
        super(Seq2SeqNMT, self).__init__()

        self.src_embeddings = nn.Embedding(
            num_embeddings=config.src_vocab_size, 
            embedding_dim=config.embed_size,
            padding_idx=config.src_padding_idx,
        )

        self.tgt_embeddings = nn.Embedding(
            num_embeddings=config.tgt_vocab_size, 
            embedding_dim=config.embed_size,
            padding_idx=config.tgt_padding_idx,
        )
        
        self.encoder = nn.LSTM(
            input_size=config.embed_size, 
            hidden_size=config.hidden_size, 
            num_layers=config.encoder_layers,
            batch_first=True, 
            bidirectional=False,
        )
        
        self.decoder = nn.LSTM(
            input_size=config.embed_size, 
            hidden_size=config.hidden_size, 
            num_layers=config.decoder_layers,
            batch_first=True, 
            bidirectional=False,
        )
        
        self.fc = nn.Linear(config.hidden_size, config.tgt_vocab_size)

    
    def forward(self, inputs):
        src_input_ids = inputs["encoder_input_ids"] # [bsz, src_len]
        src_embeddings = self.src_embeddings(src_input_ids) # [bsz, src_len, embed_size]
        src_hidden_states, (src_last_hidden_state, _) = self.encoder(src_embeddings)

        tgt_input_ids = inputs["decoder_input_ids"] # [bsz, tgt_len]
        tgt_embeddings = self.tgt_embeddings(tgt_input_ids)

        h_0 = src_last_hidden_state
        c_0 = torch.zeros(h_0.size()).to(h_0.device)
        tgt_hidden_states, (tgt_last_hidden_state, _) = self.decoder(tgt_embeddings, (h_0, c_0))

        logits = self.fc(tgt_hidden_states)

        outputs = {
            "logits": logits
        }

        return outputs

    def decode(self, inputs, tokenizer, new2old):
        """inputs: [bsz, seq_len, vocab_size]"""
        preds = inputs.argmax(-1) # [bsz, seq_len]
        preds = preds.numpy().tolist()
        preds = [[new2old[str(new_id)] for new_id in x] for x in preds]

        decoded_texts = [tokenizer.decode(x) for x in preds]

        return decoded_texts




        
    

