import torch
from torch import nn
from torch.nn.modules import TransformerEncoderLayer
from modules.rztx import RZTXEncoderLayer
from cvrp.config import CVRPConfig
from tsp.bq_network import dict_to_cpu


class BQPolicyNetwork(nn.Module):
    """
    Re-implementation of policy network as described in the paper
    "BQ-NCO: Bisimulated Quotienting for Efficient Neural Combinatorial Optimization"
    for CVRP.
    """
    def __init__(self, config: CVRPConfig, device: torch.device = None):
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

        # Embed all nodes into latent space. A node is a 4-dimensional vector consisting of the 2D-coordinates,
        # its demand and the current capacity of the vehicle
        self.node_embedding = nn.Linear(4, config.latent_dimension)

        # Additive position embedding to mark the depot node and the starting node.
        self.depot_start_node_marker = nn.Embedding(num_embeddings=2, embedding_dim=config.latent_dimension)

        # The encoded nodes get projected with a linear layer to their logit
        self.policy_linear_out = nn.Linear(config.latent_dimension, 2)

    def forward(self, x: dict):
        batch_size = x["nodes"].shape[0]
        num_nodes = x["nodes"].shape[1]  # including depot
        node_seq = x["nodes"]  # (batch, num_nodes, 2)
        # Concatenate the current capacity and demands to the node_seq
        node_seq = torch.concatenate([
            node_seq,
            x["demands"][:, :, None],
            x["current_capacity"][:, None, :].repeat((1, num_nodes, 1))
        ], dim=2)

        embedded_node_seq = self.node_embedding(node_seq)  # (batch, num_nodes, latent_dim)

        depot_marker = self.depot_start_node_marker(torch.tensor([0], dtype=torch.long, device=self.device))
        start_node_marker = self.depot_start_node_marker(torch.tensor([1] * batch_size, dtype=torch.long, device=self.device))
        embedded_node_seq[:, 0] = embedded_node_seq[:, 0] + depot_marker  # mark depot

        # Now mark the start node. This can be either the depot or the first node after the depot
        start_marker_for_depot_or_first = x["start_node_mask"][:, :, None].repeat((1, 1, self.config.latent_dimension)) * start_node_marker[:, None, :]  # (B, 2, latent_dim)
        embedded_node_seq[:, :2] = embedded_node_seq[:, :2] + start_marker_for_depot_or_first

        seq = embedded_node_seq
        for trf_block in self.graph_encoder:
            seq = trf_block(seq)

        logits = self.policy_linear_out(seq).view(batch_size, -1)  # two entries per node
        # remove logits for depot
        logits = logits[:, 2:]
        # apply the action mask
        logits[x["action_mask"]] = float("-inf")

        return logits

    def get_weights(self):
        return dict_to_cpu(self.state_dict())
