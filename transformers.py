import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by head"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False) 
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # (value, key, query) all have the shape: N, seq_length, embed_size
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]  # seq_length

        #split embeddings into self.head pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim) # N, seq_length, embed_size --> N, seq_length, heads, head_dim
        keys = keys.reshape(N, key_len, self.heads, self.head_dim) # N, seq_length, embed_size --> N, seq_length, heads, head_dim
        queries = keys.reshape(N, query_len, self.heads, self.head_dim) # N, seq_length, embed_size --> N, seq_length, heads, head_dim

        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # values shape: (N, values_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # after einsum (N, query_len, heads, heads_dim) flatten last two dimension
        # ----> N, query_len/seq_length, heads * heads_dim/embed_length

        out = self.fc_out(out)   # N, query_len/seq_length, heads * heads_dim/embed_length
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout()


    def forward(self, value, key, query, mask):
        # (value, key, query) all have the shape: N, seq_length, embed_size
        attention = self.attention(value, key, query, mask)     # N, seq_length, embed_size  --> N, seq_length, embed_size

        x = self.dropout(self.norm1(attention + query))    # N, seq_length, embed_size  --> N, seq_length, embed_size

        forward = self.feed_forward(x)   # N, seq_length, embed_size  --> N, seq_length, embed_size
        out = self.dropout(self.norm2(forward + x))  # N, seq_length, embed_size  --> N, seq_length, embed_size
        return out




class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.positional_encoding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout = dropout, forward_expansion=forward_expansion)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, mask):
        N, seq_length = x.shape   #INPUT SHAPE
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)  #Shape: seq_length --> N, seq_lenght

        out = self.dropout(self.word_embedding(x) + self.positional_encoding(positions))  #Shape: N, seq_length --> N, seq_length, embed_size

        for layer in self.layers:
            out = layer(out, out, out, mask)  # N, seq_length, embed_size  --> N, seq_length, embed_size

        return out




class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x,x,x, trg_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

    
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device) 
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape 
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__():
        pass

    def forward():
        pass
