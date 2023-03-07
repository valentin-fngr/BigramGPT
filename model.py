import torch.nn as nn 
import torch 
import config 
import torch.nn.functional as F


class BigramBaseline(nn.Module): 

    def __init__(self, vocab_size): 
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, idx):  
        bs, T, c = idx.shape[0], idx.shape[1], self.vocab_size
        logits = self.embeddings(idx) # (B, T, C)
        # weak mean aggregation using matrix multiplication
        mask = torch.tril(torch.ones(bs, T, T)).to(config.device) # (B, T, T)
        mask = mask / mask.sum(2, keepdim=True)
        logits = torch.bmm(mask, logits)  # (B, T, C)
        logits = logits.view(-1, c) # (-1, C)
        return logits
    
    def generate(self, max_gen_length):
        
        sequence = torch.zeros((1, 1)).to(config.device).type(torch.long)
        for i in range(max_gen_length): 
            # compute logit
            logits = self(sequence[:, -1][:, None])
            probs = F.softmax(logits, dim=1) 
            sample_next_idx = torch.multinomial(probs, 1)
            sequence = torch.concat([sequence, sample_next_idx], dim=1)

        return sequence
