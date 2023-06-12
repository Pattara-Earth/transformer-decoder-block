import time
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

# parameters config
context_length = 8
batch_size = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 32
num_heads = 4
lr = 1e-3
max_new_token = 500
epochs = 10000

url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
res = requests.get(url)
res.raise_for_status()
text = res.content.decode("utf-8")

chars = sorted(list(set(text)))
vocab_size = len(chars)

# text tokenization in character level
itos = {i:c for i, c in enumerate(chars)}
stoi = {c:i for i, c in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s] # encoder: string to list of int
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: list of int to string

data = torch.tensor(encode(text), dtype=torch.long)

# Train Test split
n = int(0.9*len(data))
train = data[:n]
test = data[n:]

def get_batch(split):
    data = train if split == 'train' else test
    ix = torch.randint(len(data) - context_length, (batch_size, ))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# single head self-attention
class Head(nn.Module):
    def __init__(self, n_embd, head_size, context_length):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        
        self.head_size = head_size
        
    def forward(self, x):
        B, T, C = x.shape # (B, T, C) C=n_embd
        k = self.key(x) # (B, T, C) @ (C, H) = (B, T, H)
        q = self.query(x) # (B, T, C) @ (C, H) = (B, T, H)
        # attention scores (affinities)
        att = q @ k.transpose(-2, -1) / self.head_size**0.5
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        att = F.softmax(att, dim=-1)
        # weighted sum of the values
        v = self.value(x) # (B, T, C) @ (C, H) = (B, T, H)
        out = att @ v
        return out
    

# multi-head self-attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd, head_size, context_length):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, context_length) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, num_heads * head_size)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    

# feedforward networks
class FeedForward(nn.Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(fan_in, 4 * fan_out),
                nn.ReLU(),
                nn.Linear(4 * fan_out, fan_out),
        )
        
    def forward(self, x):
        return self.net(x)

# decoder Transformer block
class Block(nn.Module):
    def __init__(self, num_heads, n_embd, head_size, context_length):
        super().__init__()
        self.sd_heads = MultiHeadAttention(num_heads, n_embd, head_size, context_length)
        self.ffwd = FeedForward(n_embd, n_embd)
        
    def forward(self, x):
        x = x + self.sd_heads(x)
        x = x + self.ffwd(x)
        return x
    

# bigram language model
class BigramLanguageModel(nn.Module):
    def __init__(self, 
                     vocab_size, 
                     n_embd, 
                     num_heads, # heads params
                     head_size, # heads params
                     context_length 
                ):
        super().__init__()
        self.token_embbeding_table = nn.Embedding(vocab_size, n_embd) # (B, T) -> (B, T, C)
        self.position_embbeding_table = nn.Embedding(context_length, n_embd)
        self.blocks = nn.Sequential(
                Block(num_heads, n_embd, head_size, context_length),
                Block(num_heads, n_embd, head_size, context_length),
                Block(num_heads, n_embd, head_size, context_length)
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.context_length = context_length
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embbeding_table(idx) # (B, T, C)  C=n_embd
        pos_emb = self.position_embbeding_table(torch.arange(T, device=device)) # (T, C) C=n_embd
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, C) C=vocab_size
        
        if targets is None:
            loss = None
        else:
            # change logits shape for cross entropy loss (B, C)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
        
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.concat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    

model = BigramLanguageModel(
                vocab_size=vocab_size,
                n_embd=n_embd,
                num_heads=num_heads,
                head_size=n_embd // num_heads,
                context_length=context_length
            ).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
sum(p.numel() for p in model.parameters())

for epoch in range(epochs):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
print(f'loss: {loss.item():.4f}')


out = decode(model.generate(torch.zeros((1,1), dtype=torch.long), 1000)[0].tolist())

input("Press Enter to continue...")

# # overwrite print
t = out

for i in range(1, len(t)+1):
    timefluc = torch.rand(1)*0.04
    time.sleep(timefluc.item())
    print(f"{t[:i]}", end='\r', flush=True)