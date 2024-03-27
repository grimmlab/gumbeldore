import math
import torch
from torch import nn
from torch.nn.modules import TransformerEncoderLayer
from modules.rztx import RZTXEncoderLayer
from jssp.config import JSSPConfig
from tsp.bq_network import dict_to_cpu


class JSSPPolicyNetwork(nn.Module):
    """
    Job-Shop Scheduling Network.
    It consists of a stack of Transformer Encoder Layers, through which we pass the full sequence of operations.
    Even layers are there for attention between operations of individual jobs. Odd layers then perform attention between
    operations which must run on the same machine.

    Operations are first affinely embedded, then passed through the stack of Trf blocks, and in the end project each
    encoded operation to a logit. We then gather the logits of the next operation to schedule for each job and use these
    as the distribution over the jobs (masking already finished jobs).
    """
    def __init__(self, config: JSSPConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu") if device is None else device

        self.encoder = nn.ModuleList([])
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
            self.encoder.append(block)

        if not config.use_rezero_transformer:
            self.final_trf_block = TransformerEncoderLayer(
                d_model=config.latent_dimension, nhead=config.num_attention_heads,
                dim_feedforward=config.feedforward_dimension, dropout=config.dropout,
                activation="gelu", batch_first=True, norm_first=True
            )
        else:
            self.final_trf_block = RZTXEncoderLayer(
                d_model=config.latent_dimension, nhead=config.num_attention_heads,
                dim_feedforward=config.feedforward_dimension, dropout=config.dropout,
                activation="gelu", batch_first=True
            )

        # Embed all operations into latent space. An operation is a 2-dimensional vector consisting of the
        # processing time and earliest starting time of the next operation of the job it belongs to (repeated
        # across all operations of a job)
        self.operation_embedding = nn.Linear(2, config.latent_dimension)

        # Slope for ALiBi
        self.alibi_slope = ALiBiPositionalEncoding.get_slope(config.num_attention_heads)

        # The encoded operations get projected with a linear layer to their logit
        self.policy_linear_out = nn.Linear(config.latent_dimension, 1)

    def forward(self, x: dict):
        device = x["operations"].device

        alibi_slope = self.alibi_slope.to(device)

        latent_dim = self.config.latent_dimension
        batch_size, num_jobs, num_operations = x["operations"].shape[:3]
        # Take operations, embed them. D = Latent dimension
        operations_embedded = self.operation_embedding(x["operations"])  # (B, J, O, D)
        # Add sinusoidal position embedding
        positional_encoding = self.positional_encoding(self.config.latent_dimension, length=num_operations)  # (O, D)
        operations_embedded = operations_embedded + positional_encoding[None, None, :, :]

        # Pipe the operations through the transformer blocks
        job_ops_mask = x["job_ops_mask"].view(batch_size * num_jobs, num_operations, num_operations)
        job_ops_mask = ALiBiPositionalEncoding.get_heads_attn_mask(job_ops_mask, self.config.num_attention_heads, alibi_slope)
        ops_machines_mask = x["ops_machines_mask"].repeat_interleave(self.config.num_attention_heads, dim=0)

        for i, trf_block in enumerate(self.encoder):
            if i % 2 == 0:
                # Even blocks: Operations of the individual jobs attend to the other operations within their job.
                # So we fold the job dimension into the batch dimension.
                operations_embedded = trf_block(
                    src=operations_embedded.view(batch_size * num_jobs, num_operations, latent_dim),
                    src_mask=job_ops_mask
                ).view(batch_size, num_jobs, num_operations, latent_dim)
            else:
                # Odd blocks: Operations running on same machine attend to each other
                operations_embedded = trf_block(
                    src=operations_embedded.view(batch_size, num_jobs * num_operations, latent_dim),
                    src_mask=ops_machines_mask
                ).view(batch_size, num_jobs, num_operations, latent_dim)

        # Gather the operations which need to be scheduled next
        index_tensor = x["jobs_next_op_idx"][:, :, :, None].repeat((1, 1, 1, self.config.latent_dimension))  # (B, J, 1, D)
        next_operation_seq = torch.gather(operations_embedded, dim=2, index=index_tensor).squeeze(dim=2)  # (B, J, D)

        next_operation_seq = self.final_trf_block(
            src=next_operation_seq,
            src_key_padding_mask=x["action_mask"]
        )

        logits = self.policy_linear_out(next_operation_seq)[:, :, 0]  # (B, J)
        # mask already finished jobs
        logits[x["action_mask"]] = float("-inf")

        return logits

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def positional_encoding(self, d_model: int, length: int):
        """
        Thanks to: https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        device = self.device
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model, device=device)
        position = torch.arange(0, length, device=device, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, device=device, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe


class ALiBiPositionalEncoding:
    @staticmethod
    def get_position_matrix(seq_len: int) -> torch.FloatTensor:
        """
        Returns a matrix of size (seq_len, seq_len) with the ALiBi positions which serve as the basis for
        an attn_mask. For example, for seq_len = 4, the matrix looks as follows
        [[ 0,  1,  2,  3],
         [-1,  0,  1,  2],
         [-2, -1,  0,  1],
         [-3, -2, -1,  0]]
        """
        x = torch.arange(seq_len)[None, :]
        y = torch.arange(seq_len)[:, None]
        return (x - y).float()

    @staticmethod
    def get_slope(num_heads: int) -> torch.FloatTensor:
        x = (2 ** 8) ** (1 / num_heads)
        return torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])[None, :, None, None].float()

    @staticmethod
    def get_heads_attn_mask(attn_mask: torch.FloatTensor, num_heads: int, slope: torch.FloatTensor) -> torch.FloatTensor:
        """
        Given attention mask of shape (B, seq_len, seq_len), makes an interleaved attention mask for all heads
        of shape (B * num_heads, seq_len, seq_len) with the mask multiplied in each head with the ALiBi slope.
        """
        batch_size, seq_len, _ = attn_mask.shape
        head_mask = attn_mask[:, None, :, :].repeat((1, num_heads, 1, 1))

        head_mask = head_mask * slope

        return head_mask.view(batch_size * num_heads, seq_len, seq_len)

