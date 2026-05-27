GPT_CONFIG_124M = {
    "vocab_size": 50257,        # vocabulary size
    "context_length": 1024,     # Context_length
    "emb_dim": 768,             # Embedding dimension
    "n_heads": 12,              # number of attention heads
    "n_layers": 12,             # number of layers
    "drop_rate": 0.1,           # Dropout rate
    "qkv_bais": False           # Query-Key-Value bais
}
GPT_CONFIG_MEDIUM = {
    "vocab_size": 50257,
    "context_length": 1024,     # Context_length
    "emb_dim": 1024,             # Embedding dimension
    "n_heads": 16,              # number of attention heads
    "n_layers": 24,             # number of layers
    "drop_rate": 0.1,           # Dropout rate
    "qkv_bais": False           # Query-Key-Value bais

}

GPT_CONFIG_LARGE = {
    "vocab_size": 50257,        # vocabulary size
    "context_length": 1024,     # Context_length
    "emb_dim": 1280,             # Embedding dimension
    "n_heads": 20,              # number of attention heads
    "n_layers": 36,             # number of layers
    "drop_rate": 0.1,           # Dropout rate
    "qkv_bais": False           # Query-Key-Value bais

}

GPT_CONFIG_XL = {
    "vocab_size": 50257,        # vocabulary size
    "context_length": 1024,     # Context_length
    "emb_dim": 1600,             # Embedding dimension
    "n_heads": 25,              # number of attention heads
    "n_layers": 48,             # number of layers
    "drop_rate": 0.1,           # Dropout rate
    "qkv_bais": False           # Query-Key-Value bais

}




import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(            #1 Uses a placeholder for TransformerBlock
            *[DummyTransformerBlock(cfg)            #1 Uses a placeholder for TransformerBlock
              for _ in range(cfg["n_layers"])]      #1 Uses a placeholder for TransformerBlock
        )

        self.final_norm = DummyLayerNorm(cfg["emb_dim"])        #2 Uses a placeholder for LayerNorm
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )


    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    

class DummyTransformerBlock(nn.Module):         #3 A simple placeholder class that will be replaced by a real TransformerBlock later
    def __init__(self, cfg):
        super().__init__()
        
    def forward(self, x):               #4 This block does nothing and just returns its input.
        return x


class DummyLayerNorm(nn.Module):            #5 A simple placeholder class that will be replaced by a real LayerNorm later
    def __init__(self, normalized_shape, eps=1e-5):         #6 The parameters here are just to mimic the LayerNorm interface.
        super().__init__()

    def forward(self, x):
        return x
    



import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
#print(batch)


torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
#print("Output shape:", logits.shape)
#print(logits)


torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
#print(out)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
#print("Mean:\n", mean)
#print("Variance:\n", var)


out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
#print("Normalized layer outputs:\n", out_norm)
#print("Mean:\n", mean)
#print("Variance:\n", var)

torch.set_printoptions(sci_mode=False)
#print("Mean:\n", mean)
#print("Variance:\n", var)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    

ln = LayerNorm(emb_dim=6)
out_ln = ln(out)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
#print("Mean:\n", mean)
#print("Variance:\n", var)



class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    


import matplotlib.pyplot as plt
gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)  #1
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "RELU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
#plt.show()


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
    


ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)       #1 Creates sample input with batch dimension 2
out = ffn(x)
#print(out.shape)


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([       #1  Implements five layers
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])
    
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)     #2  Compute the output of the current layer
            if self.use_shortcut and x.shape == layer_output.shape:         #3 Check if shortcut can be applied
                x = x + layer_output
            else:
                x = layer_output
        return x
    


layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)      #1 Specifies random seed for the initial weights for reproducibility
model_without_shortcuts = ExampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)

def print_gradients(model, x):
    output = model(x)       #1 Forward pass
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)     #2  Calculates loss based on how close the target and output are

    loss.backward()     #3 Backward pass to calculate the gradients

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")




#print_gradients(model_without_shortcuts, sample_input)

torch.manual_seed(123)
model_with_shortcuts = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
#print_gradients(model_with_shortcuts, sample_input)

from attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bais=cfg["qkv_bais"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):       #1 Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut        #2 Add the original input back

        shortcut = x    #3  Shortcut connection for feed forward block
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut    #4  Adds the original input back
        return x
        


torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

#print("Input Shape:", x.shape)
#print("Output Shape:", output.shape)


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
#1 The device setting will allow us to train the model on a CPU or GPU, depending on which device the input data sits on.
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    






torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
#print("Input Batch:\n", batch)
#print("\nOutput shape:", out.shape)
#print(out)

total_params = sum(p.numel() for p in model.parameters())
#print(f"Total number of parameters: {total_params:,}")

#print("Token embedding layer shape:", model.tok_emb.weight.shape)
#print("Output layer shape:", model.out_head.weight.shape)


total_params_gpt2 = (
    total_params - sum(p.numel() for p in model.out_head.parameters())
)

#print(f'Number of traniable parameters ' f"considering waieght tying: {total_params_gpt2:,}")

total_sizes_bytes = total_params * 4    #1  Calculates the total size in bytes (assuming float32, 4 bytes per parameter)
total_sizes_mb = total_sizes_bytes / (1024 * 1024)      #2 Converts to megabytes
#print(f"Total size of the model: {total_sizes_mb:.2f} MB")

for name, cfg in [
    ("GPT-2 Medium", GPT_CONFIG_MEDIUM),
    ("GPT-2 Large", GPT_CONFIG_LARGE),
    ("GPT-2 XL", GPT_CONFIG_XL)
]:
    #model = GPTModel(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    total_sizes_mb = (total_params * 4) / (1024 * 1024)
    #print(f"{name}")
    #print(f"  Total parameters: {total_params:,}")
    #print(f"        Model Size:     {total_sizes_mb:.2f} MB\n")


def generate_text_simple(model, idx, max_new_tokens, context_size):     #1 idx is a (batch, n_tokens) array of indices in the current context.
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  #2 Crops current context if it exceeds the supported context size e.g. if LLM supports only 5 tokens, and the context size is 10, then only the last 5 tokens are used as context
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]       #3  Focuses only on the last time step, so that (batch, n_token, vocab_size) becomes (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)      #4 probas has shape (batch, vocab_size).
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx_next - torch.argmax(probas, dim=-1, keepdim=True)   #5 idx_next has shape (batch, 1).
        idx = torch.cat((idx, idx_next), dim=1)     #6  Appends sampled index to the running sequence, where idx has shape (batch, n_tokens+1)
        
    return idx
    



start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
#print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) #1
#print("encoded_tensor.shape:", encoded_tensor.shape)



model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)