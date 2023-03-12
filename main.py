import torch  
import torch.nn as nn
import config
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from model import BigramBaseline, BigramAttn
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# read text 
# get set of characters 
# get char to number 
# get number to char 


torch.manual_seed(1337)

with open(config.data, "r") as f:
    text = f.read()
    chars = sorted(list(set(text)))
    stn = {char: i for i, char in enumerate(chars)}
    nts = {i: char for i, char in enumerate(chars)}


# enc/decode a sequence
encode = lambda s: [stn[char] for char in s]
decode = lambda enc: "".join([nts[num] for num in enc])


# create X, y batches
def create_batches(data): 
    idx = range(len(data) - config.chunk_size)
    x = torch.stack([data[i:i+config.chunk_size] for i in idx]) 
    y = torch.stack([data[i+1:i+config.chunk_size+1] for i in idx])
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

class CustomDataset(Dataset): 

    def __init__(self, x, y): 
        self.x = x 
        self.y = y 

    def __len__(self): 
        return self.x.shape[0]

    def __getitem__(self, idx): 
        
        sample_x = self.x[idx] 
        sample_y = self.y[idx] 

        return sample_x, sample_y



def evaluate(X_test, y_test, model, criterion):

    total_loss = 0.0
    model.eval()
    nb_batches = X_test.shape[0] // config.batch_size
    for i in range(nb_batches):
    
        # batch 
        x = X_test[config.batch_size*i:(config.batch_size*i) + config.batch_size].long()
        target = y_test[config.batch_size*i:(config.batch_size*i) + config.batch_size].long().view(-1)

        preds = model(x)
        loss = criterion(preds, target) 
        total_loss += loss.item()

    return total_loss / nb_batches
     

def main(): 
    
    run_name = f" \
     simple_bigram={config.nb_blocks}_lr={config.lr}_bs={config.batch_size}_chunk_size={config.chunk_size}_#head={config.nb_heads} \
     _dim_head={config.dim_head}\
    "

    writer = SummaryWriter("runs/" + run_name)

    X_train, X_test, y_train, y_test = get_data()
    nb_batches = len(X_train) // config.batch_size
    criterion = torch.nn.CrossEntropyLoss()
    model = BigramAttn(len(chars)).to(config.device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    print("Number of batches : ", nb_batches)
    print("number params : ", model.get_number_params())

    print("--- Training started ---", "\n")
    for epoch in range(config.epochs): 
        model.train()

        total_loss = 0.0

        for i in tqdm(range(nb_batches)):
        
        # batch 
            x = X_train[config.batch_size*i:(config.batch_size*i) + config.batch_size].long()
            target = y_train[config.batch_size*i:(config.batch_size*i) + config.batch_size].long().view(-1)
        
            preds = model(x)
            loss = criterion(preds, target) 

            with torch.no_grad():
                total_loss += loss.item()
            
            optimizer.zero_grad()    
            loss.backward() 
            optimizer.step() 

        eval_loss = evaluate(X_test, y_test, model, criterion)
        print("train : ", total_loss / nb_batches, "eval : ", eval_loss)
        writer.add_scalar("Train/Loss", total_loss / nb_batches, epoch)
        writer.add_scalar("Eval/Loss", eval_loss, epoch)
        writer.add_text(f"text_at_epoch_{epoch}", "".join(decode(model.generate(500)[0].tolist())), epoch)



if __name__ == "__main__":
    main() 
