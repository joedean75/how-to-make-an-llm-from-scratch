import torch
from arcitecture import GPTModel

#torch.set_printoptions(sci_mode=True, precision=4)

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  #1  We shorten the context length from 1,024 to 256 tokens.
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,   #2 It’s possible and common to set dropout to 0.
    "qkv_bais": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()


import tiktoken
from arcitecture import generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)     #1 .unsqueeze(0) adds the batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)     #2 Removes batch dimension    
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

#print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
#torch.manual_seed(123)

inputs = torch.tensor([[16833, 3626, 6100],         #["every effort moves"] 
                       [40, 1107, 588]])            #["I really like"]

targets = torch.tensor([[3626, 6100, 345],          #["effort moves you"] 
                        [1107, 588, 11311]])        #["really like chocalate"]

with torch.no_grad():       #1 Disables gradient tracking since we are not training yet
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)   #2 Probability of each token in vocabulary
#print(probas.shape)

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
#print("Token Ids:\n", token_ids)

#print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
#print(f"Outputs batch 1:"
      #f"{token_ids_to_text(token_ids[0].flatten(), tokenizer)}")


text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
#print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
#print("Text 2:", target_probas_2)


#BACKPROPAGATION

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
#print(log_probas)

avg_log_probas = torch.mean(log_probas)
#print(avg_log_probas)

neg_avg_log_probas = avg_log_probas * -1
#print(neg_avg_log_probas)

#print("Logits shape", logits.shape)
#print("Targets shape", targets.shape)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
#print("Flattened logits", logits_flat.shape)
#print("Flattened targets", targets_flat.shape)


loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
#print(loss)

perplexity = torch.exp(loss)


file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()


total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
#print("Characters:", total_characters)
#print("Tokens:", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


from chapter02 import create_dataloader_v1
torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)


#print("Train Loader:")
#for x,y in train_loader:
#    print(x.shape, y.shape)


#print("\nValidation Loader:")
#for x,y in val_loader:
#    print(x.shape, y.shape)


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)        #1 The transfer to a given device allows us to transfer the data to a GPU.
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)      #1  Iteratives over all batches if no fixed num_batches is specified
    else:
        num_batches = min(num_batches, len(data_loader))            #2  Reduces the number of batches to match the total number of batches in the data loader if num_batches exceeds the number of batches in the data loader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()           #3 Sums loss for each batch
        else:
            break

    return total_loss / num_batches         #4 Averages the loss over all batches  



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device) #1 1 If you have a machine with a CUDA-supported GPU, the LLM will train on the GPU without making any changes to the code.
with torch.no_grad():           #2 Disables gradient tracking for efficiency because we are not training yet
    train_loss = calc_loss_loader(train_loader, model, device)      #3 Via the “device” setting, we ensure the data is loaded onto the same device as the LLM model.
    val_loss = calc_loss_loader(val_loader, model, device)
#print("Training loss:", train_loss)
#print("Validation loss:", val_loss)


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []        #1 Initializes lists to track losses and tokens seen
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):     #2 Starts the main training loop
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()       #3 Resets loss gradients from the previous batch iteration
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()             #4 Calculates loss gradients
            optimizer.step()            #5 Updates model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:        #6 Optional evaluation step
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}):"
                      f"Train loss {train_loss:.3f},"
                      f"Val loss {val_loss:.3f}"                      
                      )
                

        generate_and_print_sample(                  #7 Prints a sample text after each epoch
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() #1 Dropout is disabled during evaluation for stable, reproducible results.
    with torch.no_grad():       #2 Disables gradient tracking, which is not required during evaluation, to reduce the computational overhead
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )

    model.train()
    return train_loss, val_loss




def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))      #1 Compact print format
    model.train()




torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),      #1The .parameters() method returns all trainable weight parameters of the model.
    lr=0.0004, weight_decay=0.1
)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=tokenizer
)


import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Valdation loss"
    )

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()       #1 Creates a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)    #2 Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()


epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
#plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"]
)
#print("Output text:\n", token_ids_to_text(token_ids, tokenizer))



vocab = { 
    "closer": 0,
    "every": 1, 
    "effort": 2, 
    "forward": 3,
    "inches": 4,
    "moves": 5, 
    "pizza": 6,
    "toward": 7,
    "you": 8,
} 
inverse_vocab = {v: k for k, v in vocab.items()}

next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[next_token_id])
torch.manual_seed(123) 
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])

def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item()
              for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")


#print_sampled_tokens(probas)

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


tempuratures = [1, 0.1, 5]          #1 Original, lower, and higher confidence
scaled_probas = [softmax_with_temperature(next_token_logits, T)
                 for T in tempuratures]

x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(tempuratures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i],
                   bar_width, label = f'Temperature = {T}')

ax.set_ylabel("Probability")
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
#plt.show()
'''
for i, T in enumerate(tempuratures):
    print(f"\nTemperatures: {T}")
    print_sampled_tokens(scaled_probas[i])
'''