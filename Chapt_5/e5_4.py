import torch
from arcitecture import GPTModel

from pretraining import train_model_simple, train_loader, val_loader, tokenizer, generate_and_print_sample

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  #1  We shorten the context length from 1,024 to 256 tokens.
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,   #2 It’s possible and common to set dropout to 0.
    "qkv_bais": False
}




device = torch.device("cpu")
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=1, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

generate_and_print_sample(model, tokenizer, device, start_context="Every effort moves you")