import re
import tiktoken

import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)       #1 Tokenizes the entire text

        for i in range(0, len(token_ids) - max_length, stride):         #2 Uses a sliding window to chunk the book into overlapping sequences of max_length
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self):          #3 Returns the total number of rows in the dataset
        return len(self.input_ids)
    
    def __getitem__(self, idx):         #4 Returns a single row from the dataset
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")   #1 Initializes the tokenizer
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)  #2 Creates dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,        #3 drop_last=True drops the last batch if it is shorter than the specified batch_size to prevent loss spikes during training.
        num_workers=num_workers     #4 The number of CPU processes to use for preprocessing
    )

    return dataloader

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text



class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i  in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]        #1 Replaces unknown words by <|unk|> tokens

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)           #2 Replaces spaces before the specified punctuations
        return text
    


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
preprocessed = result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
#print(len(preprocessed)) print for testing
#print(preprocessed[:30]) print for testing

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
#print(vocab_size) print for testing

vocab = {token:integer for integer,token in enumerate(all_words)}
#for i, item in enumerate(vocab.items()):
    #print(item)
    #if i >= 50:
        #break

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know." Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
#print(ids)
#print(tokenizer.decode(ids))

text="Hello, do you like tea?"
#print(tokenizer.encode(text)) #Hello cuase keyword error


all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

#print(len(vocab.items()))

#for i, item in enumerate(list(vocab.items())[-5:]):
    #print(item)




text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = "<|endoftext|> ".join((text1, text2))

#print(text)

tokenizer = SimpleTokenizerV2(vocab)
#print(tokenizer.encode(text))
#print(tokenizer.decode(tokenizer.encode(text)))

tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces""of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
#print(integers)
strings = tokenizer.decode(integers)
#print(strings)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
#print(len(enc_text))



enc_sample = enc_text[50:]

context_size = 4    #1 The context size determines how many tokens are included in the input.
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
#print(f"x: {x}")
#print(f"y:      {y}")


for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    #print(context, "---->", desired)


for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    #print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))



#with open("the-verdict.txt", "r", encoding="utf-8") as f:
 #   raw_text = f.read

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)
data_iter = iter(dataloader)        #1 Converts dataloader into a Python iterator to fetch the next entry via Python’s built-in next() function
#first_batch = next(data_iter)
#print(first_batch)

#second_batch = next(data_iter)
#print(second_batch)

inputs, targets = next(data_iter)
#print("Inputs:\n", inputs)
#print("\nTargets:\n", targets)

input_ids = torch.tensor([2, 3, 5, 1])

vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

#print(embedding_layer.weight)
#print(embedding_layer(torch.tensor([3])))
#print(embedding_layer(input_ids))

vocab_size = 50257
output_dim = 256

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, 
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

#print("Token IDs:\n", inputs)
#print("\nInupts shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
#print(token_embeddings.shape)

# uncomment & execute the following to see how the embeddings look like
#print(token_embeddings)

context_length = max_length

pos_embedding_layer = torch.nn.Embedding(context_size, output_dim)

# uncomment & execute the following line to see how the embedding layer weights look like
#print(pos_embedding_layer.weight)

pos_embeddings = pos_embedding_layer(torch.arange(max_length))
#print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
#print(input_embeddings.shape)

# uncomment & execute the following line to see how the embeddings look like
#print(input_embeddings)