import torch.nn as nn 
import torch 
import config 
import torch.nn.functional as F


class BigramBaseline(nn.Module): 

    def __init__(self, vocab_size, d_emb=256): 
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_emb)
        self.positional_embedding = nn.Embedding(config.chunk_size, d_emb)
        
        self.linear = nn.Linear(d_emb, vocab_size)
        self.vocab_size = vocab_size
        self.d_emb = d_emb

    def forward(self, idx):  
        bs, T, d = idx.shape[0], idx.shape[1], self.d_emb
        logits = self.embeddings(idx) # (B, T, d)
        positions = self.positional_embedding(torch.arange(T,device=config.device)[None, :]) # (B, T, d)
        
        x = logits + positions # (B, T, d)

        # weak mean aggregation using matrix multiplication
        # mask = torch.tril(torch.ones(bs, T, T)).to(config.device) # (B, T, T)
        # mask = mask / mask.sum(2, keepdim=True)
        # logits = torch.bmm(mask, logits)  # (B, T, d)

        output = self.linear(x) # (B, T, d)
        output = output.view(-1, self.vocab_size)
        return output
    
    def generate(self, max_gen_length):
        with torch.no_grad():

            sequence = torch.zeros((1, 1)).to(config.device).type(torch.long)
            for i in range(max_gen_length): 
                # compute logit
                logits = self(sequence[:, -1][:, None])
                probs = F.softmax(logits, dim=1) 
                sample_next_idx = torch.multinomial(probs, 1)
                sequence = torch.concat([sequence, sample_next_idx], dim=1)

            return sequence
