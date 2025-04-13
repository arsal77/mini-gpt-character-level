
import sys
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import yaml
import argparse

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
context_len = config.get('context_len', 128)
n_emb = config.get('n_emb', 384)
n_heads = config.get('n_heads', 6)
n_layers = config.get('n_layers', 6)

default_prompt = config.get('start_prompt', '\n')
default_max_tokens = config.get('max_gen_tokens', 500)

model_load_path = config.get('model_load_path', 'model/mini_gpt_model.pth')

parser = argparse.ArgumentParser(description="Generate text using a pre-trained MiniGPT model (loads most config from config.yaml).")
parser.add_argument('--prompt', type=str, default=default_prompt,
                    help=f'Starting prompt for text generation (default: "{default_prompt}")')
parser.add_argument('--length', type=int, default=default_max_tokens,
                    help=f'Maximum number of tokens to generate (default: {default_max_tokens})')

args = parser.parse_args()

final_prompt = args.prompt
final_max_tokens = args.length

full_corpus_path = os.path.join(project_root, corpus_path)
full_model_load_path = os.path.join(project_root, model_load_path)

head_size = n_emb // n_heads

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(346322)

if device == 'cuda':
    torch.cuda.manual_seed(346322)

text = load_data(full_corpus_path)
vocab_size, stoi, itos,enc_data = encode_data(text)

model = MiniGPT(
        vocab_size=vocab_size,
        n_emb=n_emb,
        head_size=head_size,
        n_heads=n_heads,
        context_len=context_len,
        n_layers=n_layers,
        stoi=stoi,
        itos=itos,
        encode=encode,
        decode=decode
    )
model.to(device)
print("Model architecture created.")

state_dict = torch.load(full_model_load_path, map_location=device)
model.load_state_dict(state_dict)
print("Model weights loaded successfully.")

print(f"\n--- Generating {final_max_tokens} tokens starting with: ---")
print(final_prompt)
print("-----------------------------------------------")

generated_text = model.generate(final_prompt, max_sample_gen=final_max_tokens)
print(generated_text)

print("-----------------------------------------------")
print("Generation complete.")