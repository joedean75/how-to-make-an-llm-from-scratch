import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],    # Your    (x^1)
     [0.55, 0.87, 0.66],    # journey (x^2)
     [0.57, 0.85, 0.64],    # starts  (x^3)
     [0.22, 0.58, 0.33],    # with    (x^4)
     [0.77, 0.25, 0.10],    # one     (x^5)
     [0.05, 0.80, 0.55]]    #step     (x^6)
)



# (Lines 14-79) Simple self-attention mechanism

query = inputs[1]   #1 The second input token serves as the query.
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
#print(attn_scores_2)

res = 0
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
#print(res)
#print(torch.dot(inputs[0], query))

# Normalization
att_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
#print("Attention weight:", att_weights_2_tmp)
#print("Sum", att_weights_2_tmp.sum())

# Softmax function for normalization 
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

att_weights_2_naive = softmax_naive(attn_scores_2)
#print("Attention weight:", att_weights_2_naive)
#print("Sum", att_weights_2_naive.sum())

# pytorch implemantion of softmax (Best Practice)
att_weights_2 = torch.softmax(attn_scores_2, dim=0)
#print("Attention weight:", att_weights_2)
#print("Sum", att_weights_2.sum())


query = inputs[1]   #1 The second input token is the query.
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += att_weights_2[i]*x_i
#print(context_vec_2)


#compute all context vectors
#Each element in the tensor represents an attention score between each pair of inputs
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] =torch.dot(x_i, x_j)
#print(attn_scores)


#matrix multiplication will do the same as the above for loop but faster
attn_scores = inputs @ inputs.T
#print(attn_scores)

#Normalize each row so the values add up to 1
att_weights = torch.softmax(attn_scores, dim=-1)
#print(att_weights)

#verify rows add up to 1
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
#print("Row 2 sum:", row_2_sum)
#print("All row sums:", att_weights.sum(dim=-1))

# use attention weights to compute all context vectors with matrix multiplication
all_context_vecs = att_weights @ inputs
#print(all_context_vecs)
#print("Previous 2nd context vector:", context_vec_2)


#(Lines 82-) Self attention Mechanism a.k.a Scaled dot-product attention
# we want to compute context vectors as weighted sums over the input vectors specific to a certain input element


#Step by step

#W(q) = query
#W(k) = key
#W(v) = value vectors

# single context vector for example
x_2 = inputs[1]             #1 The second input element
d_in = inputs.shape[1]      #2 The input embedding size, d=3
d_out = 2                   #3 The output embedding size, d_out=2

#requires_grad set to false for example to reduce clutter, normally true to update the matrices while training model
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
#print(query_2)


#obtain all keys and values via matrix multiplication

keys = inputs @ W_key
values = inputs @ W_value
#print("keys.shape:", keys.shape)
#print("values.shape:", values.shape)


#First, let’s compute the attention score ω(22)
keys_2 = keys[1]        #1 remeber that Python strats indexing at 0
attn_scores_22 = query_2.dot(key_2)
#print(attn_scores_22)

# we can generalize this computation to all attention scores via matrix multiplication
attn_scores_2 = query_2 @ keys.T    #1 All attention scores for given query
#print(attn_scores_2)

#compute attention weights by scaling the attention scores and using the softmax function
d_k = keys.shape[-1]
att_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
#print(att_weights_2)

#compute context vectors
context_vec_2 = att_weights_2 @ values
#print(context_vec_2)


#Self contained Self Attention mechanism class for repeated use fo above example
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T # omega
        #print(attn_scores)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        #print(attn_weights)

        context_vec = attn_weights @ values 
        return context_vec
    

'''
The __init__ method initializes trainable weight matrices (W_query, W_key, and W_value) for queries, keys, and values, each transforming the input dimension d_in to an output dimension d_out.

During the forward pass, using the forward method, we compute the attention scores (attn_scores) by multiplying queries and keys, normalizing these scores using softmax. Finally, we create a context vector by weighting the values with these normalized attention scores.
'''

#validating above class
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
#print(sa_v1(inputs))


#Self atentinon class optimized using nn.linear from pytorch
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bais=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bais)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bais)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bais)
        
    
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        #print(attn_scores)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        #print(att_weights)
        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
#print(sa_v2(inputs))


#d_in, d_out = 3, 2

#sa_v1 = SelfAttention_v1(d_in, d_out)
#sa_v2 = SelfAttention_v2(d_in, d_out)

# Copy v2 weights → v1, transposing to match the shape difference
#sa_v1.W_query = nn.Parameter(sa_v2.W_query.weight.T)
#sa_v1.W_key   = nn.Parameter(sa_v2.W_key.weight.T)
#sa_v1.W_value = nn.Parameter(sa_v2.W_value.weight.T)

# Both should now produce identical outputs
#print(sa_v1(inputs))
#print(sa_v2(inputs))

queries = sa_v2.W_query(inputs)     #1 Reuses the query and key weight matrices of the SelfAttention_v2 object from the previous section for convenience
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
#print(attn_weights)

#using pytorch's tril function to mask values above the diagonal
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
#print(mask_simple)

#multiply mask by weihgts to zero out above the diagonal
masked_simple = attn_weights*mask_simple
#print(masked_simple)

#normalize the wieghts to sum up to 1 in each row
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
#print(masked_simple_norm)


mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
#print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
#print(attn_weights)

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)     #1 We choose a dropout rate of 50%.
example = torch.ones(6, 6)          #2 Here, we create a matrix of 1s
#print(dropout(example))

torch.manual_seed(123)
#print(dropout(attn_weights))


batch = torch.stack((inputs, inputs), dim=0)
#print(batch.shape)            #1 Two inputs with six tokens each; each token has embedding dimension 3.

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)      #1 Compared to the previous SelfAttention_v1 class, we added a dropout layer.
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )       #2 The register_buffer call is also a new addition (more information is provided in the following text).
    
    def forward(self, x):
        d, num_tokens, d_in = x.shape           #3 We transpose dimensions 1 and 2, keeping the batch dimension at the first position (0).
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)            #4 In PyTorch, operations with a trailing underscore are performed in-place, avoiding unnecessary memory copies.
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec



torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
#print("context_vecs.shape:", context_vecs.shape)


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bais=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(
                    d_in, d_out, context_length, dropout, qkv_bais
                )
            for _ in range(num_heads)]
        )
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    

torch.manual_seed(123)
context_length = batch.shape[1]         # This is the number of tokens
d_in, d_out = 3, 1
'''mha = MultiHeadAttentionWrapper(
    d_in, d_out, context_length, 0.0, num_heads=2
)'''
#context_vecs = mha(batch)
#print(context_vecs)
#print("context_vecs.shape:", context_vecs.shape)



class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0),  "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads          #1 Reduces the projection dim to match the desired output dim
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)         #2  Uses a Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)    #3 Tensor shape: (b, num_tokens, d_out)
        queries = self.W_query(x)   #3 Tensor shape: (b, num_tokens, d_out)
        values = self.W_value(x)   #3 Tensor shape: (b, num_tokens, d_out)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)      #4 We implicitly split the matrix by adding a num_heads dimension. Then we unroll the last dim: (b, num_tokens, d_out) -&gt; (b, num_tokens, num_heads, head_dim).
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)      #5Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)   #5 Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)     #5 Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)

        attn_scores = queries @ keys.transpose(2, 3)    #6  Computes dot product for each head
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  #7 Masks truncated to the number of tokens
        attn_scores.masked_fill_(mask_bool, -torch.inf)       #8 Uses the mask to fill attention scores
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)   #9 Tensor shape: (b, num_tokens, n_heads, head_dim)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )       #10 Combines heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = self.out_proj(context_vec)        #11 Adds an optional linear projection
        return context_vec
        



torch.manual_seed(123)

d_in = 768
d_out = 768
context_length = 1024
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=12)
#context_vecs = mha(batch)
#print(context_vecs)
#print("context_vecs.shape:", context_vecs.shape)

# batch shape: (batch_size, num_tokens, d_in)

dummy_batch = torch.randn(2, 1024, 768)
context_vecs = mha(dummy_batch)
#print("context_vecs.shape:", context_vecs.shape)  # expect: torch.Size([2, 1024, 768])
#print(context_vecs)

#export MultiHeadAttention()