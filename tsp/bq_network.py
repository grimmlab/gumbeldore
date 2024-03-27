import torch
from torch import nn
from torch.nn.modules import TransformerEncoderLayer
from modules.rztx import RZTXEncoderLayer
from tsp.config import TSPConfig


class BQPolicyNetwork(nn.Module):
    """
    Re-implementation of Policy network as described in the paper
    "BQ-NCO: Bisimulated Quotienting for Efficient Neural Combinatorial Optimization"
    """
    def __init__(self, config: TSPConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu") if device is None else device

        self.graph_encoder = nn.ModuleList([])
        for _ in range(config.num_transformer_blocks):
            if not config.use_rezero_transformer:
                block = TransformerEncoderLayer(
                    d_model=config.latent_dimension, nhead=config.num_attention_heads,
                    dim_feedforward=config.feedforward_dimension, dropout=config.dropout,
                    activation="gelu", batch_first=True, norm_first=True
                )
            else:
                block = RZTXEncoderLayer(
                    d_model=config.latent_dimension, nhead=config.num_attention_heads,
                    dim_feedforward=config.feedforward_dimension, dropout=config.dropout,
                    activation="gelu", batch_first=True
                )
            self.graph_encoder.append(block)

        # Embed all nodes into latent space
        self.node_embedding = nn.Linear(2, config.latent_dimension)

        # Additive marker to indicate first/last node (not masked if at least one node has been chosen)
        self.first_last_marker = nn.Embedding(num_embeddings=2, embedding_dim=config.latent_dimension)

        # The encoded nodes get projected with a linear layer to their logit
        self.policy_linear_out = nn.Linear(config.latent_dimension, 1)

    def forward(self, x: dict):
        batch_size = x["nodes"].shape[0]
        node_seq = x["nodes"]
        embedded_node_seq = self.node_embedding(node_seq)  # (batch, num_nodes, latent_dim)

        first_marker = self.first_last_marker(torch.tensor([0] * batch_size, dtype=torch.long, device=self.device))
        last_marker = self.first_last_marker(torch.tensor([1] * batch_size, dtype=torch.long, device=self.device))

        embedded_node_seq[:, 0] = embedded_node_seq[:, 0] + first_marker  # mark first node
        embedded_node_seq[:, -1] = embedded_node_seq[:, -1] + last_marker  # mark last node

        seq = embedded_node_seq
        for trf_block in self.graph_encoder:
            seq = trf_block(seq)

        logits = self.policy_linear_out(seq).squeeze(-1)
        # remove logit for start and destination
        logits = logits[:, 1:-1]

        return logits

    def get_weights(self):
        return dict_to_cpu(self.state_dict())


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict
