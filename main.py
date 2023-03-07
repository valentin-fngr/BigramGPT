import torch  
import torch.nn as nn
import config
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import BigramBaseline

# read text 
# get set of characters 
# get char to number 
# get number to char 


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




def main(): 

    X_train, X_test, y_train, y_test = get_data()
    nb_batches = X_train.shape[0] // config.batch_size
    criterion = torch.nn.CrossEntropyLoss()
    model = BigramBaseline(len(chars)).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    print("Number of batches : ", nb_batches)

    print("--- Training started ---", "\n")
    for epoch in range(config.epochs): 

        total_loss = 0.0

        for i in range(nb_batches):
            
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

    
        print(f"Epoch {epoch} : cross entropy = {total_loss / nb_batches}")
        # print("".join(decode(model.generate(300)[0].tolist())))

if __name__ == "__main__":
    main() 
