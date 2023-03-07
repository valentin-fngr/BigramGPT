import torch.nn as nn 
import torch 
import config 
import torch.nn.functional as F




class AttentionHead(nn.Module): 

    def __init__(self, in_c, d_head): 
        super().__init__()
        self.query = nn.Linear(in_c, d_head, bias=False) 
        self.key = nn.Linear(in_c, d_head, bias=False) 
        self.value = nn.Linear(in_c, d_head, bias=False)  
        self.d_head = d_head

    def forward(self, x):
        """
            x : (B, T, in_c)
        """ 

        q  = self.query(x) # (B, T, d_head)
        k = self.key(x) # (B, T, d_head)
        v = self.value(x) # (B, T, d_head)

        attn = (q@k.transpose(2, 1)) / (self.d_head)**(1/2) # (B, T, T)
        # compute softmax for the query against the key
        prob_attn = F.softmax(attn, dim=1) # (B, T, T) 
        output = prob_attn @ v  # (B, T, D) 
        return output
        

class MultiHeadAttention(nn.Module): 


    def __init__(self, in_c, nb_head, d_out):
        
        if d_out % nb_head != 0: 
            raise ValueError("d_out must be divisible by nb_head")
        
        super().__init__()
        self.multi_head = nn.ModuleList([AttentionHead(in_c, d_out // nb_head) for i in range(nb_head)])

    def forward(self, x): 
        output = torch.concat([head(x) for head in self.multi_head], dim=-1)
        return output


class BigramBaseline(nn.Module): 

    def __init__(self, vocab_size, d_emb=25): 
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




class BigramAttn(nn.Module): 

    def __init__(self, vocab_size, d_emb=256): 
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_emb)
        self.positional_embedding = nn.Embedding(config.chunk_size, d_emb)
        self.attn_head1 = MultiHeadAttention(d_emb, 8, d_emb)
        self.attn_head2 = MultiHeadAttention(d_emb, 8, d_emb)
        
        self.linear = nn.Linear(d_emb, vocab_size)
        self.vocab_size = vocab_size
        self.d_emb = d_emb

    def forward(self, idx):  
        bs, T, d = idx.shape[0], idx.shape[1], self.d_emb
        logits = self.embeddings(idx) # (B, T, d)
        positions = self.positional_embedding(torch.arange(T,device=config.device)[None, :]) # (B, T, d)
        
        x = logits + positions # (B, T, d)
        x = self.attn_head1(x) # (B, T, d)
        x = self.attn_head2(x)
        output = self.linear(x) # (B, T, vocab_size)
        output = output.view(-1, self.vocab_size)
        return output
    
    def generate(self, max_gen_length):
        with torch.no_grad():

            sequence = torch.zeros((1, 1)).to(config.device).type(torch.long)
            for i in range(max_gen_length): 
                # compute logit
                logits = self(sequence[:, -config.chunk_size:])
                probs = F.softmax(logits, dim=1) 
                sample_next_idx = torch.multinomial(probs, 1)
                sequence = torch.concat([sequence, sample_next_idx[-1][None, :]], dim=1)
            print(sequence)
            return sequence
