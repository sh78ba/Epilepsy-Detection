import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalConv1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=self.padding,
            dilation=dilation
        )
        
    def forward(self, x):
        x = self.conv(x)
        # Remove future information (right side padding)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection - 1x1 conv if input/output channels differ
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual path
        res = x if self.residual is None else self.residual(x)
        
        # Ensure same length for residual connection
        if res.size(2) != out.size(2):
            res = res[:, :, :out.size(2)]
        
        # Add residual
        out = out + res
        out = self.relu2(out)
        
        return out


class TCN(nn.Module):
    
    def __init__(self, input_channels, num_channels, kernel_size=4, dropout=0.2):
        super(TCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, ...
            in_ch = input_channels if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            
            layers.append(
                ResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class SelfAttention(nn.Module):
    
    def __init__(self, embed_dim, dropout=0.1):
        super(SelfAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.scale = math.sqrt(embed_dim)
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        # Transpose to (batch, length, channels) for attention
        x = x.transpose(1, 2)
        batch_size, seq_len, embed_dim = x.size()
        
        # Compute Q, K, V
        Q = self.query(x)  # (batch, seq_len, embed_dim)
        K = self.key(x)
        V = self.value(x)
        
        # Scaled dot-product attention
        # Attention weights = softmax(Q * K^T / sqrt(d_k))
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch, seq_len, embed_dim)
        
        # Transpose back to (batch, channels, length)
        attn_output = attn_output.transpose(1, 2)
        
        return attn_output


class TCN_SA(nn.Module):
    
    def __init__(self, 
                 input_channels=1,
                 input_length=100,
                 num_tcn_channels=[64, 64],  # 2 blocks with 64 channels each
                 kernel_size=4,
                 num_classes=2,
                 tcn_dropout=0.2,
                 sa_dropout=0.1,
                 fc_dropout=0.3):
        super(TCN_SA, self).__init__()
        
        # TCN blocks
        self.tcn = TCN(
            input_channels=input_channels,
            num_channels=num_tcn_channels,
            kernel_size=kernel_size,
            dropout=tcn_dropout
        )
        
        # Self-Attention
        self.self_attention = SelfAttention(
            embed_dim=num_tcn_channels[-1],
            dropout=sa_dropout
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Fully Connected Classifier
        self.fc1 = nn.Linear(num_tcn_channels[-1], 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        
        # TCN
        x = self.tcn(x)
        
        # Self-Attention
        x = self.self_attention(x)
        
        # Global Average Pooling
        x = self.gap(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)
        
        # Classifier
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(input_channels=1, input_length=100, num_tcn_blocks=2, 
                 tcn_channels=128, kernel_size=4, num_classes=2):
    
    num_tcn_channels = [tcn_channels] * num_tcn_blocks
    
    model = TCN_SA(
        input_channels=input_channels,
        input_length=input_length,
        num_tcn_channels=num_tcn_channels,
        kernel_size=kernel_size,
        num_classes=num_classes
    )
    
    print(f"Model created with {model.count_parameters():,} parameters")
    
    return model


if __name__ == "__main__":
    # Test model creation and parameter count
    print("=" * 60)
    print("Testing TCN-SA Model")
    print("=" * 60)
    
    # Create model
    model = create_model()
    
    # Test with dummy input
    batch_size = 4
    input_channels = 1
    input_length = 100
    
    x = torch.randn(batch_size, input_channels, input_length)
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.count_parameters():,}")
    
    print("\n" + "=" * 60)
    print("Model test completed successfully!")
    print("=" * 60)
