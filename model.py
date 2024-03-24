class ConvEmbed(nn.Module):
    def __init__(self, prior_channels, embed_size, kernel_size, stride, padding):
        super().__init__()

        self.proj = nn.Conv2d(prior_channels, embed_size, kernel_size, stride, padding)
        self.embednorm = nn.LayerNorm(embed_size)

    def forward(self, x):
        emb = self.proj(x)
        bs, c, h, w = emb.shape
        emb = emb.view(bs, h*w, c)
        emb = self.embednorm(emb)
       
        return emb 
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, pointwise_kernel, pointwise_stride, bias=False):
        super().__init__()

        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = self.embed_size // self.num_heads
        self.dropout = nn.Dropout(dropout)

        assert embed_size % num_heads == 0, 'Embedding size should be divisible to number of heads'

        self.conv_proj_q = nn.Sequential(
            nn.Conv2d(embed_size, embed_size, pointwise_kernel, pointwise_stride, bias=bias),
            nn.BatchNorm2d(embed_size)
        )
        self.conv_proj_k = nn.Sequential(
            nn.Conv2d(embed_size, embed_size, pointwise_kernel, pointwise_stride, bias=bias),
            nn.BatchNorm2d(embed_size)
        )
        self.conv_proj_v = nn.Sequential(
            nn.Conv2d(embed_size, embed_size, pointwise_kernel, pointwise_stride, bias=bias),
            nn.BatchNorm2d(embed_size)
        )

        self.linear_proj_q = nn.Linear(embed_size, embed_size, bias=bias)
        self.linear_proj_k = nn.Linear(embed_size, embed_size, bias=bias)
        self.linear_proj_v = nn.Linear(embed_size, embed_size, bias=bias)

    def forward(self, x):
        bs, hw, emb = x.shape
        x = x.view(bs, emb, int(np.sqrt(hw)), int(np.sqrt(hw)))
        

        q_conv = self.conv_proj_q(x)
        k_conv = self.conv_proj_k(x)
        v_conv = self.conv_proj_v(x)

        bs, ch, h, w = q_conv.shape

        q_conv = q_conv.view(bs, h*w, ch)
        k_conv = k_conv.view(bs, h*w, ch)
        v_conv = v_conv.view(bs, h*w, ch)

        q = self.linear_proj_q(q_conv)
        k = self.linear_proj_k(k_conv)
        v = self.linear_proj_v(v_conv)

        q = q.view(bs, self.num_heads, -1, self.head_dim)
        k = k.view(bs, self.num_heads, -1, self.head_dim)
        v = v.view(bs, self.num_heads, -1, self.head_dim)

        scores = q @ k.transpose(-2, -1) * self.head_dim ** -0.5
        scores = F.softmax(scores, dim=-1)
        output = scores @ v
        output = output.view(bs, -1, ch)
        

        return self.dropout(output)


class FeedForward(nn.Module):
    def __init__(self, embed_size , dropout=dropout):
        super().__init__()

        self.ffn = nn.Sequential(
        nn.Linear(embed_size, embed_size * 4),
        nn.GELU(),
        nn.Linear(embed_size * 4, embed_size),
        nn.Dropout(dropout))

    def forward(self, x):
        x = self.ffn(x)
        return x 
    
class Block(nn.Module):
    def __init__(self, embed_size, num_heads, pointwise_kernel, pointwise_stride, dropout=dropout):
        super().__init__()
        self.mha = MultiHeadAttention(embed_size, num_heads, pointwise_kernel, pointwise_stride)
        self.layernorm1 = nn.LayerNorm(embed_size)
        self.ffn = FeedForward(embed_size, dropout)
        self.layernorm2 = nn.LayerNorm(embed_size)
        
    def forward(self, x):
        x = x + self.mha(x)
        x = x + self.ffn(self.layernorm1(x))
        x = self.layernorm2(x)
        return x
    
class CvT(nn.Module):
    def __init__(self):
        super().__init__()

        self.convembed_stage1 = ConvEmbed(stage1_prior, stage1_embed, stage1_kernel, stage1_stride, stage1_padding)
        self.blocks_stage1 = nn.Sequential(
            *[Block(embed_size=stage1_embed, num_heads=stage1_heads,pointwise_kernel=pointwise_kernel, pointwise_stride=pointwise_stride, dropout=dropout) for _ in range(stage1_num_blocks)]
        )
        self.convembed_stage2 = ConvEmbed(stage2_prior, stage2_embed, stage2_kernel, stage2_stride, stage2_padding)
        self.blocks_stage2 = nn.Sequential(
            *[Block(embed_size=stage2_embed, num_heads=stage2_heads,pointwise_kernel=pointwise_kernel, pointwise_stride=pointwise_stride, dropout=dropout) for _ in range(stage2_num_blocks)]
        )
        self.convembed_stage3 = ConvEmbed(stage3_prior, stage3_embed, stage3_kernel, stage3_stride, stage3_padding)
        self.blocks_stage3 = nn.Sequential(
            *[Block(embed_size=stage3_embed, num_heads=stage3_heads,pointwise_kernel=pointwise_kernel, pointwise_stride=pointwise_stride, dropout=dropout) for _ in range(stage3_num_blocks)]
        )
        
        
        self.finalmlp = nn.Linear(stage3_embed, n_classes, bias=False)
    
    def forward(self, x):
        x = self.convembed_stage1(x)
        x = self.blocks_stage1(x)
        bs, hw, emb = x.shape
        x = x.view(bs, emb, int(np.sqrt(hw)), int(np.sqrt(hw)))
        x = self.convembed_stage2(x)
        x = self.blocks_stage2(x)
        bs, hw, emb = x.shape
        x = x.view(bs, emb, int(np.sqrt(hw)), int(np.sqrt(hw)))
        x = self.convembed_stage3(x)
        x = self.blocks_stage3(x)
        
        output = self.finalmlp(x[:, 0])
        return output