import torch


class PatchedEmbed(torch.nn.Module):

    # layer that is responsible for patching
    # patching is done through the unfold + linear

    def __init__(
        self, patch_h: int, patch_w: int, in_channels: int, embedding_dim: int
    ):
        super().__init__()

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.unfold = torch.nn.Unfold(
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        self.projection = torch.nn.Linear(
            self.in_channels * self.patch_h * self.patch_w, self.embedding_dim
        )

    def forward(self, x) -> torch.Tensor:
        patches = self.unfold(x).swapdims(-1, -2)
        return self.projection(patches)


class PatchEmbedConv(torch.nn.Module):

    # layer that is responsible for patching
    # patching is done through the convolution operation

    def __init__(
        self, patch_h: int, patch_w: int, in_channels: int, embedding_dim: int
    ):
        super().__init__()

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.conv = torch.nn.Conv2d(
            self.in_channels,
            self.embedding_dim,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
            padding=0,
        )

    def forward(self, x) -> torch.Tensor:
        return self.conv(x).flatten(start_dim=-2, end_dim=-1).swapdims(-1, -2)


class PositionEmbed(torch.nn.Module):
    def __init__(self, seq_len: int, embedding_dim: int):
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

        self.embedding = torch.nn.Parameter(
            torch.empty((self.seq_len + 1, self.embedding_dim))
        )
        torch.nn.init.xavier_normal_(self.embedding)

    def forward(self, x):
        return self.embedding


class SelfAttentionBlock(torch.nn.Module):

    # inspired by https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
    # https://einops.rocks/pytorch-examples.html

    def __init__(self, embedding_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.dropout = dropout

        self.qkv = torch.nn.Linear(self.embedding_dim, 3 * self.embedding_dim)
        self.res_dropout = torch.nn.Dropout(dropout)
        self.out_proj = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, -1)

        batch_size = x.size(0)
        embed_dim = x.size(-1)
        head_dim = embed_dim // self.n_heads

        # sdpa expects input shape n, ..., num_heads, seq_len, head_dim
        q = q.view(batch_size, -1, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, head_dim).transpose(1, 2)

        if self.training:
            dropout = self.dropout
        else:
            dropout = 0.0

        attn_output = (
            torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=dropout, is_causal=False
            ).transpose(1, 2)
            # if dropout is bigger than zero then we need contiguous() here or the next view is going to fail
            .contiguous()
        )

        out_projected = self.out_proj(attn_output.view(batch_size, -1, embed_dim))
        return self.res_dropout(out_projected)


class MLPBlock(torch.nn.Module):
    def __init__(
        self, embedding_in: int, embedding_out: int, hidden_size: int, dropout_p: float
    ):
        super().__init__()

        self.embedding_in = embedding_in
        self.embedding_out = embedding_out
        self.hidden_size = hidden_size

        self.dropout = torch.nn.Dropout(dropout_p)

        self.linear1 = torch.nn.Linear(self.embedding_in, self.hidden_size)
        self.linear2 = torch.nn.Linear(self.hidden_size, self.embedding_out)

        self.activation = torch.nn.GELU()

    def forward(self, x) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ViTBlock(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        dropout: float,
        mlp_hidden: int,
        embedding_out: int,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.mlp_hidden = mlp_hidden
        self.embedding_out = embedding_out

        self.ln1 = torch.nn.LayerNorm(self.embedding_dim)

        self.attn = SelfAttentionBlock(self.embedding_dim, self.n_heads, self.dropout)

        self.ln2 = torch.nn.LayerNorm(self.embedding_dim)

        self.mlp = MLPBlock(
            self.embedding_dim, self.embedding_out, self.mlp_hidden, self.dropout
        )

    def forward(self, x):
        attn_res = self.attn(self.ln1(x)) + x
        mlp_output = self.mlp(self.ln2(attn_res))
        return attn_res + mlp_output


class ViT(torch.nn.Module):

    # n blocks and preprocess and postprocess

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

        # plan
        # x -> create patches -> embed -> multihead -> feedforward -> output -> sort of
