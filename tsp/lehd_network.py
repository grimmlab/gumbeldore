import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from tsp.config import TSPConfig
from tsp.bq_network import dict_to_cpu


class LEHDPolicyNetwork(nn.Module):
    """
    Re-implementation of Policy network as described in the paper
    "Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization"
    """
    def __init__(self, config: TSPConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu") if device is None else device

        # Embed all nodes into latent space
        self.node_embedding = nn.Linear(2, config.latent_dimension)
        self.graph_encoder = NoNormTransformerEncoderLayer(
                d_model=config.latent_dimension, nhead=config.num_attention_heads,
                dim_feedforward=config.feedforward_dimension, dropout=config.dropout,
                activation="gelu", batch_first=True
            )

        self.graph_decoder = nn.ModuleList([])
        for _ in range(config.num_transformer_blocks):
            block = NoNormTransformerEncoderLayer(
                d_model=config.latent_dimension, nhead=config.num_attention_heads,
                dim_feedforward=config.feedforward_dimension, dropout=config.dropout,
                activation="gelu", batch_first=True
            )
            self.graph_decoder.append(block)

        # Affine projection of first/last node
        self.start_projection = nn.Linear(config.latent_dimension, config.latent_dimension)
        self.dest_projection = nn.Linear(config.latent_dimension, config.latent_dimension)

        # The encoded nodes get projected with a linear layer to their logit
        self.policy_linear_out = nn.Linear(config.latent_dimension, 1)

    def encode(self, nodes: torch.Tensor):
        """
        Parameters:
            nodes [torch.Tensor]: _All_ nodes of shape (batch, num nodes, 2)

        Returns:
            [torch.Tensor] Encoded nodes of shape (batch, num nodes, latent dim).
        """
        embedded_node_seq = self.node_embedding(nodes)
        return self.graph_encoder(embedded_node_seq)

    def decode(self, nodes: torch.Tensor):
        """
        Parameters:
            nodes [torch.Tensor]: Nodes corresponding to subtour to plan of shape
            (batch, num nodes in subtour, 2). We expect the start node to be in the
            first position and the destination node in the last position of the sequence.

        Returns:
            [torch.Tensor] Logits of nodes in subtour (excluding start and end)
        """
        start_node = self.start_projection(nodes[:, 0:1])
        dest_node = self.dest_projection(nodes[:, -1:])
        # we need to concatenate back in this way because we cannot in-place
        # modify the first and last element due to autograd.
        nodes = torch.cat((start_node, nodes[:, 1:-1], dest_node), dim=1)

        seq = nodes
        for trf_block in self.graph_decoder:
            seq = trf_block(seq)

        logits = self.policy_linear_out(seq).squeeze(-1)
        # remove logit for start and destination
        logits = logits[:, 1:-1]
        return logits

    def forward(self, x: dict):
        """
        Full pass through encoder and decoder as used in training. See `RandomTSPDataset`
        for how the dictionary should be formed.
        """
        encoded_nodes = self.encode(x["nodes"])
        subtour_nodes = encoded_nodes[:, :x["subtour_length"]]

        if x["start_node_is_end_node"]:
            # concat start node additionally to the end
            subtour_nodes = torch.cat((subtour_nodes, subtour_nodes[:, 0:1]), dim=1)

        return self.decode(subtour_nodes)

    def get_weights(self):
        return dict_to_cpu(self.state_dict())


class NoNormTransformerEncoderLayer(nn.Module):
    """
    Regular TransformerEncoderLayer without any normalization as used in the paper
    "Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization"
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu', batch_first=False):
        super().__init__()

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in PyTroch Transformer class.
        """
        # Self attention layer
        src2 = src
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2 = src2[0] # no attention weights
        src = src + self.dropout1(src2)

        # Pointwise FF Layer
        src2 = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src