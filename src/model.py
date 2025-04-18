
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Head(nn.Module) :
  def __init__(self,n_emb,head_size,context_len) :
    super().__init__()
    self.query = nn.Linear(n_emb,head_size,bias = False)
    self.key = nn.Linear(n_emb,head_size,bias = False)
    self.value = nn.Linear(n_emb,head_size,bias=False)
    self.dropout = nn.Dropout(0.2)
    self.head_size=head_size
    self.register_buffer('tril',torch.tril(torch.ones(context_len,context_len)))

  def forward(self,idx) :
    B,T,C = idx.shape
    query = self.query(idx) 
    key = self.key(idx) 
    value = self.value(idx) 
    wei = query@key.transpose(-1,-2) 
    wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
    wei = wei*(self.head_size**-0.5) 
    wei = F.softmax(wei,dim=1)
    wei = self.dropout(wei)

    out = wei@value 

    return out

class MultiHeadedAttention(nn.Module) :
  def __init__(self,n_emb,head_size,n_heads,context_len) :
    super().__init__()
    self.multi_head = nn.ModuleList(Head(n_emb,head_size,context_len) for _ in range(n_heads))
    self.proj = nn.Linear(head_size*n_heads,n_emb,bias = True)
    self.dropout = nn.Dropout(0.2)


  def forward(self,idx) :
    out = torch.cat([m(idx) for m in self.multi_head],dim = -1)
    out = self.proj(out)
    out = self.dropout(out)

    return out

class FeedForward(nn.Module) :
  def __init__(self,n_emb) :
    super().__init__()
    self.ff = nn.Sequential(nn.Linear(n_emb,n_emb*4),nn.ReLU(),nn.Linear(n_emb*4,n_emb), nn.Dropout(0.2))

  def forward(self,idx) :
    return self.ff(idx)

class Block(nn.Module) :
  def __init__(self,n_emb,head_size,n_heads,context_len) :
    super().__init__()
    self.multi_head = MultiHeadedAttention(n_emb,head_size,n_heads,context_len)
    self.ff = FeedForward(n_emb)
    self.layer_norm_1 = LayerNorm1d(n_emb)
    self.layer_norm_2 = LayerNorm1d(n_emb)


  def forward(self,idx) :
    out = idx + self.multi_head(self.layer_norm_1(idx))
    out = out + self.ff(self.layer_norm_2(out))
    return out

class LayerNorm1d(nn.Module):
    def __init__(self,dim,eps=1e-5) :
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1,dim).to(device))
        self.beta = nn.Parameter(torch.zeros(1,dim).to(device))
        self.eps = eps


    def forward(self,x) :

        xmean = x.mean(-1,keepdim = True)
        xvar = x.var(-1,keepdim = True)
        xhat = (x-xmean)/torch.sqrt(xvar+self.eps)
        self.out = self.gamma*xhat + self.beta

        return self.out


    

class MiniGPT(nn.Module) :

  def __init__(self,vocab_size,n_emb,head_size,n_heads,context_len,stoi,itos,encode,decode,n_layers):
    super().__init__()
    self.token_embedding_lookup = nn.Embedding(vocab_size,n_emb)
    self.position_embedding_lookup = nn.Embedding(context_len,n_emb)
    self.block = nn.Sequential(*[Block(n_emb,head_size,n_heads,context_len) for i in range(n_layers)])
    self.layer_norm = LayerNorm1d(n_emb)
    self.lm_head = nn.Linear(head_size*n_heads,vocab_size)
    self.decode = decode
    self.context_len = context_len
    self.vocab_size = vocab_size
    self.apply(self._init_weights)
    self.encode=encode
    self.stoi = stoi
    self.itos=itos

  def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self,idx,target = None) :

    B,T = idx.shape
    emb_token = self.token_embedding_lookup(idx) 
    emb_pos = self.position_embedding_lookup(torch.arange(T,device=device)) 
    emb = emb_token + emb_pos
    x=self.block(emb)
    x=self.layer_norm(x)
    logits = self.lm_head(x) 

    if target is not None:
      logits= logits.view(-1,self.vocab_size)
      target = target.view(target.shape[0]*target.shape[1])
      loss = F.cross_entropy(logits,target)
    else :
      loss = None
    return logits,loss

  @torch.no_grad()
  def generate(self,text,max_sample_gen):
    self.eval()
    idx = torch.unsqueeze(torch.tensor(self.encode(text,self.stoi)),0).to(device=device)
    for _ in range(max_sample_gen):
      logits,loss = self(idx[:,-self.context_len:])
      probs = F.softmax(logits[:,-1,:],dim=-1)
      next_token = torch.multinomial(probs,1)
      idx = torch.cat([idx,next_token],dim=1)
    self.train()
    return ''.join(self.decode(torch.squeeze(idx,0).tolist(),self.itos))