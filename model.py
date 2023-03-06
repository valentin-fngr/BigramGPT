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
        bs, T, c = config.batch_size, config.chunk_size, self.vocab_size
        # if bs = 2, and we use chunk size of 8 
        # it will return 2 batches of 8 vectors of dim 10
        logits = self.embeddings(idx).view(-1, c)
        return logits
    
    def generate(self, max_gen_length):
        
        sequence = torch.zeros((1, 1)).to(config.device).type(torch.long)
        for i in range(max_gen_length): 
            # compute logit
            logits = self(sequence[:, -1])
            probs = F.softmax(logits, dim=1) 
            sample_next_idx = torch.multinomial(probs, 1)
            sequence = torch.concat([sequence, sample_next_idx], dim=1)

        return sequence
