# from typing import Any, Dict, List, Optional

# import torch
# from torch import Tensor
# from torch.nn import Embedding, ModuleDict
# from torch_frame.data.stats import StatType
# from torch_geometric.data import HeteroData
# from torch_geometric.nn import MLP
# from torch_geometric.typing import NodeType

# from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder


# class Model(torch.nn.Module):

#     def __init__(
#         self,
#         data: HeteroData,
#         col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
#         num_layers: int,
#         channels: int,
#         out_channels: int,
#         aggr: str,
#         norm: str,
#         # List of node types to add shallow embeddings to input
#         shallow_list: List[NodeType] = [],
#         # ID awareness
#         id_awareness: bool = False,
#         readout_mode: Optional[str] = None,
#     ):
#         super().__init__()

#         self.encoder = HeteroEncoder(
#             channels=channels,
#             node_to_col_names_dict={
#                 node_type: data[node_type].tf.col_names_dict
#                 for node_type in data.node_types
#             },
#             node_to_col_stats=col_stats_dict,
#         )
#         self.temporal_encoder = HeteroTemporalEncoder(
#             node_types=[
#                 node_type for node_type in data.node_types if "time" in data[node_type]
#             ],
#             channels=channels,
#         )
#         self.gnn = HeteroGraphSAGE(
#             node_types=data.node_types,
#             edge_types=data.edge_types,
#             channels=channels,
#             aggr=aggr,
#             num_layers=num_layers,
#         )
#         self.head = MLP(
#             channels,
#             out_channels=out_channels,
#             norm=norm,
#             num_layers=1,
#         )
#         self.embedding_dict = ModuleDict(
#             {
#                 node: Embedding(data.num_nodes_dict[node], channels)
#                 for node in shallow_list
#             }
#         )

#         self.id_awareness_emb = None
#         if id_awareness:
#             self.id_awareness_emb = torch.nn.Embedding(1, channels)
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.encoder.reset_parameters()
#         self.temporal_encoder.reset_parameters()
#         self.gnn.reset_parameters()
#         self.head.reset_parameters()
#         for embedding in self.embedding_dict.values():
#             torch.nn.init.normal_(embedding.weight, std=0.1)
#         if self.id_awareness_emb is not None:
#             self.id_awareness_emb.reset_parameters()

#     def forward(
#         self,
#         batch: HeteroData,
#         entity_table: NodeType,
#     ) -> Tensor:
#         seed_time = batch[entity_table].seed_time
#         x_dict = self.encoder(batch.tf_dict)

#         rel_time_dict = self.temporal_encoder(
#             seed_time, batch.time_dict, batch.batch_dict
#         )

#         for node_type, rel_time in rel_time_dict.items():
#             x_dict[node_type] = x_dict[node_type] + rel_time

#         for node_type, embedding in self.embedding_dict.items():
#             x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

#         x_dict = self.gnn(
#             x_dict,
#             batch.edge_index_dict,
#             batch.num_sampled_nodes_dict,
#             batch.num_sampled_edges_dict,
#         )

#         return self.head(x_dict[entity_table][: seed_time.size(0)])

#     def forward_dst_readout(
#         self,
#         batch: HeteroData,
#         entity_table: NodeType,
#         dst_table: NodeType,
#     ) -> Tensor:
#         if self.id_awareness_emb is None:
#             raise RuntimeError(
#                 "id_awareness must be set True to use forward_dst_readout"
#             )
#         seed_time = batch[entity_table].seed_time
#         x_dict = self.encoder(batch.tf_dict)
#         # Add ID-awareness to the root node
#         x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

#         rel_time_dict = self.temporal_encoder(
#             seed_time, batch.time_dict, batch.batch_dict
#         )

#         for node_type, rel_time in rel_time_dict.items():
#             x_dict[node_type] = x_dict[node_type] + rel_time

#         for node_type, embedding in self.embedding_dict.items():
#             x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

#         x_dict = self.gnn(
#             x_dict,
#             batch.edge_index_dict,
#         )

#         return self.head(x_dict[dst_table])

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder


class Model(torch.nn.Module):
    """Heterogeneous GNN with configurable depth-wise skip/read-out mechanisms.

    readout_mode options
    --------------------
    1. "logit_sum"        – per-depth heads map features→logits, logits summed.
    2. "gated_logit_sum"  – same, but summed with learned softmax weights.
    3. "feat_cat_mlp"     – concatenate features from all depths then a shared MLP.
    """

    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        shallow_list: List[NodeType] = [],
        id_awareness: bool = False,
        # New -----------------------------------------------------------------
        readout_mode: Optional[str] = None,
        final_dropout: float = 0.0,
    ):
        super().__init__()

        assert readout_mode in {None, "logit_sum", "gated_logit_sum", "feat_cat_mlp"}, (
            "readout_mode must be None, 'logit_sum', 'gated_logit_sum', or 'feat_cat_mlp'"
        )
        self.readout_mode: Optional[str] = readout_mode
        self.final_dropout = final_dropout
        self.num_layers = num_layers
        self.channels = channels
        self.out_channels = out_channels

        # ---------------------------------------------------------------------
        # Encoders
        # ---------------------------------------------------------------------
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                nt: data[nt].tf.col_names_dict for nt in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )

        # ---------------------------------------------------------------------
        # Depth-wise read-out heads (only if skip connections enabled)
        if self.readout_mode is None:
            # Baseline: a single head applied to the final GNN layer output
            self.head = MLP(
                channels, out_channels=out_channels, norm=norm, num_layers=1
            )
        else:
            # For modes operating in logit space we need one head per depth (input + each layer).
            self.heads = torch.nn.ModuleList(
                [MLP(channels, out_channels=out_channels, norm=norm, num_layers=1)
                 for _ in range(num_layers + 1)]
            )
            # Learnable gating for "gated_logit_sum"
            self.depth_logits_weight = torch.nn.Parameter(torch.zeros(num_layers + 1))
            # Shared MLP for concatenated features (feat_cat_mlp)
            self.readout_mlp = MLP(
                (num_layers + 1) * channels,           # Input dimension (concat of all depths)
                hidden_channels=channels,              # Hidden layer size
                out_channels=out_channels,
                norm=norm,
                num_layers=2,
            )

        # Optional shallow embeddings per node type
        self.embedding_dict = ModuleDict({
            node: Embedding(data.num_nodes_dict[node], channels) for node in shallow_list
        })

        # Optional ID-awareness
        self.id_awareness_emb: Optional[torch.nn.Embedding] = None
        if id_awareness:
            self.id_awareness_emb = torch.nn.Embedding(1, channels)

        self.reset_parameters()

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        if self.readout_mode is None:
            self.head.reset_parameters()
        else:
            for head in self.heads:
                head.reset_parameters()
            self.readout_mlp.reset_parameters()
            with torch.no_grad():
                self.depth_logits_weight.zero_()
        for emb in self.embedding_dict.values():
            torch.nn.init.normal_(emb.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def _encode_inputs(
        self,
        batch: HeteroData,
        add_id_awareness_for: Optional[NodeType],
        seed_time: Optional[Tensor],
    ) -> Dict[NodeType, Tensor]:
        """Apply frame/temporal/shallow (+ optional ID-awareness) encodings."""
        x_dict = self.encoder(batch.tf_dict)

        # ID awareness only on seed slice
        if add_id_awareness_for is not None and self.id_awareness_emb is not None:
            assert seed_time is not None
            n_seed = seed_time.size(0)
            x_dict[add_id_awareness_for][:n_seed] += self.id_awareness_emb.weight

        # Temporal encodings
        if seed_time is not None:
            rel_time_dict = self.temporal_encoder(
                seed_time, batch.time_dict, batch.batch_dict
            )
            for nt, rel in rel_time_dict.items():
                x_dict[nt] = x_dict[nt] + rel

        # Shallow embeddings
        for nt, emb in self.embedding_dict.items():
            x_dict[nt] = x_dict[nt] + emb(batch[nt].n_id)

        return x_dict

    def _collect_layerwise_reps(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict,
    ) -> List[Dict[NodeType, Tensor]]:
        """Return list of representations at depths 0..L (input, then after each layer)."""
        hidden: List[Dict[NodeType, Tensor]] = [x_dict]
        for conv, norm_dict in zip(self.gnn.convs, self.gnn.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: norm_dict[k](v) for k, v in x_dict.items()}
            x_dict = {k: v.relu() for k, v in x_dict.items()}
            hidden.append(x_dict)
        return hidden

    # ------------------------------------------------------------------
    # Read-out helpers
    # ------------------------------------------------------------------
    def _readout_node_task(self, hidden, entity_table: NodeType, seed_n: int) -> Tensor:
        if self.readout_mode is None:
            return self.head(hidden[-1][entity_table][:seed_n])

        if self.readout_mode == "logit_sum":
            logits = 0
            for depth, x_d in enumerate(hidden):
                z = self.heads[depth](x_d[entity_table][:seed_n])
                z = F.dropout(z, p=self.final_dropout, training=self.training)
                logits = logits + z
            return logits

        if self.readout_mode == "gated_logit_sum":
            weights = self.depth_logits_weight.softmax(dim=0)
            logits = 0
            for depth, x_d in enumerate(hidden):
                z = self.heads[depth](x_d[entity_table][:seed_n])
                z = F.dropout(z, p=self.final_dropout, training=self.training)
                logits = logits + weights[depth] * z
            return logits

        # feat_cat_mlp
        x_cat = torch.cat([x_d[entity_table][:seed_n] for x_d in hidden], dim=-1)
        return self.readout_mlp(x_cat)

    def _readout_dst_task(self, hidden, dst_table: NodeType) -> Tensor:
        if self.readout_mode is None:
            return self.head(hidden[-1][dst_table])

        if self.readout_mode == "logit_sum":
            logits = 0
            for depth, x_d in enumerate(hidden):
                z = self.heads[depth](x_d[dst_table])
                z = F.dropout(z, p=self.final_dropout, training=self.training)
                logits = logits + z
            return logits

        if self.readout_mode == "gated_logit_sum":
            weights = self.depth_logits_weight.softmax(dim=0)
            logits = 0
            for depth, x_d in enumerate(hidden):
                z = self.heads[depth](x_d[dst_table])
                z = F.dropout(z, p=self.final_dropout, training=self.training)
                logits = logits + weights[depth] * z
            return logits

        # feat_cat_mlp
        x_cat = torch.cat([x_d[dst_table] for x_d in hidden], dim=-1)
        return self.readout_mlp(x_cat)

    # ------------------------------------------------------------------
    # Public forward(s)
    # ------------------------------------------------------------------
    def forward(self, batch: HeteroData, entity_table: NodeType) -> Tensor:
        seed_time = batch[entity_table].seed_time
        seed_n = seed_time.size(0)

        x0 = self._encode_inputs(batch, add_id_awareness_for=None, seed_time=seed_time)
        hidden = self._collect_layerwise_reps(x0, batch.edge_index_dict)
        return self._readout_node_task(hidden, entity_table, seed_n)

    def forward_dst_readout(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        dst_table: NodeType,
    ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError("id_awareness must be set True to use forward_dst_readout")

        seed_time = batch[entity_table].seed_time
        x0 = self._encode_inputs(batch, add_id_awareness_for=entity_table, seed_time=seed_time)
        hidden = self._collect_layerwise_reps(x0, batch.edge_index_dict)
        return self._readout_dst_task(hidden, dst_table)
