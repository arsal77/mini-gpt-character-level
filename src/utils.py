import torch
def load_data(corpus_path) :
    try :
      with open(corpus_path,'r',encoding='utf-8') as f:
        text = f.read()
    except FileNotFoundError :
      print(f"File not found at {corpus_path}")

    except Exception as e:
        print(f"Error reading data file (check encoding?): {e}")
        raise

    print(f"Loaded {len(text)} characters from {corpus_path}")
    if not text:
        raise ValueError("Data file seems empty.")
    return text

def encode_data(text) :
  vocab = ''.join(sorted(list(set(text))))
  vocab_len= len(vocab)
  stoi = {j:i for i,j in enumerate(vocab)}
  itos = {j:i for i,j in stoi.items()}
  encode = torch.tensor([stoi[x] for x in data],device=device)



  return vocab_len,stoi,itos,encode

def encode(text,stoi) :
  return [stoi[id] for id in text]

def decode(idx,itos) :
  return [itos[id] for id in idx]

def train_test_split(data,split=0.9) :
  X_train = data[:int(split*len(data))]
  X_val = data[int(split*len(data)):]

  return X_train,X_val

def generate_batch(data,batch_size=32,context_len=8) :
  idx = torch.randint(len(data)-context_len,(batch_size,),device=device)
  x = torch.stack([data[id:id+context_len] for id in idx],dim=0)
  y = torch.stack([data[id+1:id+context_len+1] for id in idx],dim=0)
  x=x.to(device)
  y=y.to(device)

  return x,y