import torch  
import torch.nn as nn
import config
from torch.utils.data import DataLoader
import torch.nn.functional as F

# read text 
# get set of characters 
# get char to number 
# get number to char 
with open(config.data, "r") as f:
    text = f.read()
    chars = sorted(list(set(text)))
    stn = {char: i for i, char in enumerate(chars)}
    nts = {i: char for i, char in enumerate(chars)}


print("len vocab : ", len(chars))
print("".join(chars))

# enc/decode a sequence
encode = lambda s: [stn[char] for char in s]
decode = lambda enc: "".join([nts[num] for num in enc])


chunk_size = 8

# create X, y batches
def create_batches(data): 
    idx = range(len(data) - chunk_size)
    x = torch.stack([data[i:i+chunk_size] for i in idx]) 
    y = torch.stack([data[i+1:i+chunk_size+1] for i in idx])
    return x,y


def get_data(): 

    data = torch.tensor(encode(text), dtype=torch.float32)
    train_data = data[:int(0.9*data.shape[0])]
    test_data = data[int(0.9*data.shape[0]):]

    X_train, y_train = create_batches(train_data) 
    X_test, y_test = create_batches(test_data) 

    if config.device == "cuda": 
        X_train = X_train.to(config.device)
        X_test = X_test.to(config.device)
        y_train = y_train.to(config.device)
        y_test = y_test.to(config.device)

    return X_train, X_test, y_train, y_test


        

X_train, X_test, y_train, y_test = get_data()
        


class BigramBaseline(nn.Module): 


    def __init__(self, vocab_size): 
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, idx):  
        bs, T, c = config.batch_size, chunk_size, self.vocab_size
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




model = BigramBaseline(len(chars))
model = model.to(config.device)

batch = X_train[:config.batch_size].long()

pred = model(batch)

batch_y = y_train[:config.batch_size].view(-1).long()


generated_sequence = "".join(decode(model.generate(600)[0].tolist()))
print(generated_sequence)
