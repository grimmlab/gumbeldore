import time
import torch
from torch import nn
from cvrp.config import CVRPConfig
from tsp.lehd_network import dict_to_cpu, NoNormTransformerEncoderLayer


class LEHDPolicyNetwork(nn.Module):
    """
    Re-implementation of policy network as described in the paper
    "Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization"
    """
    def __init__(self, config: CVRPConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu") if device is None else device

        # Embed all nodes into latent space. A node is a 3-dimensional vector consisting of the 2D-coordinates,
        # and its normalized demand (which is set to 0 for the depot)
        self.node_embedding = nn.Linear(3, config.latent_dimension)
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

        # Affine projection of depot node and the start node (as in paper, all subtours are expected to end at depot)
        self.depot_projection = nn.Linear(config.latent_dimension, config.latent_dimension)
        self.start_projection = nn.Linear(config.latent_dimension, config.latent_dimension)
        self.capacity_embedding = nn.Linear(1, config.latent_dimension)

        # The encoded nodes get projected with a linear layer to their logit
        self.policy_linear_out = nn.Linear(config.latent_dimension, 2)

    def encode(self, nodes: torch.Tensor, demands: torch.Tensor):
        """
        Parameters:
            nodes: (batch_size, num_nodes, 2), coordinate tensor, where first entry is depot.
            demands: (batch_size, num_nodes, 1), normalized demands, where first entry is 0.
        Returns:
            [torch.Tensor] Encoded node sequence of shape (batch, num_nodes, latent_dim)
        """
        node_seq = torch.concatenate([
            nodes,
            demands[:, :, None]
        ], dim=2)

        embedded_node_seq = self.node_embedding(node_seq)
        depot = self.depot_projection(embedded_node_seq[:, 0:1])
        embedded_node_seq = torch.cat(
            (depot, embedded_node_seq[:, 1:]), dim=1
        )
        return self.graph_encoder(embedded_node_seq)

    def decode(self, encoded_node_seq: torch.Tensor, x: dict):
        """
        Parameters:
             encoded_node_seq [torch.Tensor]: Encoded node sequence as returned from `encode`
             x [dict]: Data dictionary as obtained from the RandomCVRPDataset or Trajectory.trajectories_to_batch.
        Returns:
            [torch.Tensor] Logits of nodes in subtour (excluding depot node) where we have two logits
                for each node (reached via depot or not)
        """
        batch_size = encoded_node_seq.shape[0]
        # start_node_mask has shape (B, 2, 1) with a 1 at 0 if we start at depot else with a 1 at 1 if we have start node
        start_node_mask = x["start_node_mask"][:, :, None]
        # we project both depot and first node in the encoded sequence and then
        # the vector at position 0 or 1 will be set to:
        # mask * projected_vector + (1-mask) * original vector
        depot_and_first_node = encoded_node_seq[:, :2]
        depot_and_first_node_projected = self.start_projection(depot_and_first_node)
        encoded_node_seq[:, :2] = start_node_mask * depot_and_first_node_projected + (
                    1. - start_node_mask) * depot_and_first_node

        # add embedding of current capacity to all nodes
        current_capacity = self.capacity_embedding(x["current_capacity"])[:, None, :]  # (B, 1, latent_dim)
        encoded_node_seq = encoded_node_seq + current_capacity

        seq = encoded_node_seq
        for trf_block in self.graph_decoder:
            seq = trf_block(seq)

        logits = self.policy_linear_out(seq).view(batch_size, -1)  # two entries per node
        # remove logits for depot
        logits = logits[:, 2:]
        # apply the action mask
        logits[x["action_mask"]] = float("-inf")

        return logits

    def forward(self, x: dict):
        encoded_node_seq = self.encode(x["nodes"], x["demands"])
        return self.decode(encoded_node_seq, x)

    def get_weights(self):
        return dict_to_cpu(self.state_dict())
