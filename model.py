import torch.nn as nn 
import torch 
import config 
import torch.nn.functional as F



class FeedForward(nn.Module): 


    def __init__(self, d_in, d_out, dropout=0.2): 
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False) 
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        out = self.dropout(out)
        return out

class AttentionBlock(nn.Module): 

    def __init__(self, in_c, nb_head, d_out): 
        super().__init__()
        self.multihead = MultiHeadAttention(in_c, nb_head, d_out)
        self.ff = FeedForward(d_out, d_out)
        self.layer_norm1 = nn.LayerNorm(d_out)
        self.layer_norm2 = nn.LayerNorm(d_out)

    def forward(self, x): 
        out = self.multihead(x) 
        x = self.layer_norm1(x + out)
        out = self.ff(x)
        out = self.layer_norm2(x + out)
        return out
        

class AttentionHead(nn.Module): 

    def __init__(self, in_c, d_head): 
        super().__init__()
        self.query = nn.Linear(in_c, d_head, bias=False) 
        self.key = nn.Linear(in_c, d_head, bias=False) 
        self.value = nn.Linear(in_c, d_head, bias=False)  
        self.d_head = d_head
        self.register_buffer("mask", torch.ones(config.chunk_size, config.chunk_size, device=config.device))


    def forward(self, x):
        """
            x : (B, T, in_c)
        """ 

        q  = self.query(x) # (B, T, d_head)
        k = self.key(x) # (B, T, d_head)
        v = self.value(x) # (B, T, d_head)

        attn = (q@k.transpose(-2, -1)) / (self.d_head)**(1/2) # (B, T, T)
        attn = attn @ self.mask.masked_fill(self.mask == 0, float("-inf"))

        # compute softmax for the query against the key
        prob_attn = F.softmax(attn, dim=1) # (B, T, T) 
        output = prob_attn @ v  # (B, T, D) 
        return output
        

class MultiHeadAttention(nn.Module): 

    def __init__(self, in_c, nb_head, d_out):

        super().__init__()
        
        if d_out % nb_head != 0: 
            raise ValueError("d_out must be divisible by nb_head")        
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

    def __init__(self, vocab_size, nb_blocks=config.nb_blocks, d_emb=config.dim_head): 
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_emb)
        self.positional_embedding = nn.Embedding(config.chunk_size, d_emb)
        self.nb_blocks = nn.Sequential(*[AttentionBlock(d_emb, config.nb_heads, d_emb) for _ in range(nb_blocks)])
        
        self.linear = nn.Linear(d_emb, vocab_size)
        self.vocab_size = vocab_size
        self.d_emb = d_emb

    def forward(self, idx):  
        bs, T, d = idx.shape[0], idx.shape[1], self.d_emb
        logits = self.embeddings(idx) # (B, T, d)
        positions = self.positional_embedding(torch.arange(T,device=config.device)[None, :]) # (B, T, d)
        
        x = logits + positions # (B, T, d)
        x1 = self.nb_blocks(x) # (B, T, d)
        output = self.linear(x1) # (B, T, vocab_size)
        output = output.view(-1, self.vocab_size)
        return output
    
    def generate(self, max_gen_length):
        with torch.no_grad():

            sequence = torch.zeros((1, config.chunk_size)).to(config.device).type(torch.long)
            for i in range(max_gen_length): 
                # compute logit
                logits = self(sequence[:, -config.chunk_size:])
                probs = F.softmax(logits, dim=1) 
                sample_next_idx = torch.multinomial(probs, 1)
                sequence = torch.concat([sequence, sample_next_idx[-1][None, :]], dim=1)
            print(sequence)
            return sequence
