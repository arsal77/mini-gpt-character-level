
import sys
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import yaml


project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from model import MiniGPT
from utils import *

CONFIG_PATH = 'config.yaml'

try :
  with open('config.yaml','r') as f:
    config = yaml.safe_load(f)
except FileNotFoundError :
  print(f"Error: Configuration file not found at {CONFIG_PATH}")
  exit()
except Exception as e :
  print(f"Error loading configuration file: {e}")
  exit()

corpus_path = config.get('corpus_path', 'data/sample.txt')
train_split = config.get('split', 0.9)

# Model parameters
context_len = config.get('context_len', 128)
n_emb = config.get('n_emb', 384)
n_heads = config.get('n_heads', 6)
n_layers = config.get('n_layers', 6)

# Training parameters
batch_size = config.get('batch_size', 64)
train_iterations = config.get('train_iterations', 5000)
learning_rate = config.get('learning_rate', 3e-4)

full_corpus_path = os.path.join(project_root, corpus_path)

relative_save_path = config.get('model_load_path', 'model/mini_gpt_model.pth')
model_save_path = os.path.join(project_root, relative_save_path)
print(f"Saving final model state dictionary to: {model_save_path}")
model_save_dir = os.path.dirname(model_save_path)
os.makedirs(model_save_dir, exist_ok=True)

head_size = n_emb // n_heads
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(146432)

if device == 'cuda':
    torch.cuda.manual_seed(146432)

text = load_data(full_corpus_path)
vocab_size,stoi,itos,enc_data=encode_data(text)
enc_data.to(device)
x_train,x_val=train_test_split(enc_data,0.9)

model = MiniGPT(vocab_size,n_emb,head_size,n_heads,context_len,stoi,itos,encode,decode,n_layers)
model = model.to(device)
opt = AdamW(model.parameters(),lr=learning_rate)
scaler = GradScaler()


for i in range(train_iterations) :

  x_train_batch,y_train_batch = generate_batch(x_train,batch_size,context_len)
  x_train_batch, y_train_batch = x_train_batch.to(device), y_train_batch.to(device) 

  with autocast(enabled=(device.type == 'cuda')): 
      logits,loss=model(x_train_batch,y_train_batch)
  opt.zero_grad()
  scaler.scale(loss).backward()
  scaler.step(opt)
  scaler.update()
  model.eval()
  with torch.no_grad():  # Disable gradient calculation during validation
      x_val_batch, y_val_batch = generate_batch(x_val, batch_size, context_len)
      x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
      with autocast(enabled=(device.type == 'cuda')): # FIX: Add autocast wrapper
          logits_val, loss_val = model(x_val_batch, y_val_batch)
  model.train()

  if i%500 == 0 :
    print(f'-train--loss : {loss.item()},\n -val--loss : {loss_val.item()}')

print('training_done')

os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print("Model saved successfully.")