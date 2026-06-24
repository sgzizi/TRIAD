# coding: utf-8
r"""
TRIAD: Resolvability-Typed Adaptive Intent Hierarchies for Multimodal Recommendation
====================================================================================

Reference implementation of the model described in

    "TRIAD: Resolvability-Typed Adaptive Intent Hierarchies for Multimodal
     Recommendation."

The code is organized to mirror the methodology section of the paper. The
mapping between paper components and the classes below is:

    Paper component (section)                         ->  Class / method
    ------------------------------------------------------------------------
    Multimodal graph backbone (Sec. IV-D)             ->  CollaborativeGraphConv,
                                                          ContentGraphConv,
                                                          ModalityProjection,
                                                          GraphConvLayer
    Neighborhood-coupled flow audit / within-item     ->  FlowDispersionSensor
      dispersion w_i  (Sec. IV-B, IV-E)                     (+ ConditionalVectorField)
    Stable residual-quantized intent hierarchy        ->  ResidualQuantizer
      (Sec. IV-F)
    Resolvability-typed adaptive intent depth         ->  AdaptiveIntentHierarchy
      (Sec. IV-F, IV-G)
    Depth-attention aggregation  u_u^{(d)}  (Sec. IV-G)->  DepthAttentionAggregator
    Full model / prediction & optimization (Sec. IV-H)->  TRIAD

Built on top of the MMRec toolbox (https://github.com/enoche/MMRec).
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric
import pdb
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization
import math

torch.autograd.set_detect_anomaly(True)



class ConditionalVectorField(nn.Module):
    r"""Conditional vector field v_theta(s, x_s, h_i) of the neighborhood-coupled
    flow audit (Sec. IV-E). Given a flow time ``s``, a point ``x_s`` on the
    interpolant path, and the collaborative context ``cond`` (=h_i), it predicts
    the velocity ``x_1 - x_0`` that transports the base noise to the item's
    multimodal neighborhood distribution."""

    def __init__(self, input_dim, cond_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        initial_layer = nn.Linear(input_dim + cond_dim, hidden_dim)
        self.initial_layer = initial_layer

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.final_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, t, x, cond):
        t_emb = self.time_mlp(t.unsqueeze(-1))
        x_cond = torch.cat([x, cond], dim=-1)

        hidden = self.initial_layer(x_cond)

        # Incorporate time embedding and residual connections
        for layer in self.layers:
            hidden = hidden + layer(hidden + t_emb)

        return self.final_layer(hidden)



class FlowDispersionSensor(nn.Module):
    r"""Flow-based within-item dispersion sensor (Sec. IV-B and IV-E).

    A per-modality conditional flow is matched to the empirical distribution of
    an item's collaborative neighborhood (the neighborhood-coupled flow audit,
    L_FM). ``served_dispersion`` then reads off the within-item dispersion
    w_i for the visual and textual modalities, the irreducible ``W_u`` term of
    the three-way variance decomposition (Sec. IV-C). A cross-modal synergy
    term (weight ``lambda_cross``) couples the two modality flows and exposes
    the between-modality gap ``G_u``."""

    def __init__(self, feat_dim, cond_dim, hidden_dim, lambda_cross=1.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.cond_dim = cond_dim
        self.lambda_cross = lambda_cross

        # Conditional flows for the visual and textual modalities
        self.v_flow_net = ConditionalVectorField(feat_dim, cond_dim, hidden_dim)
        self.t_flow_net = ConditionalVectorField(feat_dim, cond_dim, hidden_dim)

        # Projection for cross-modal (between-modality gap) alignment
        self.text_to_visual_proj = nn.Linear(feat_dim, feat_dim)

    def _flow_matching_loss(self, flow_net, x_1, cond):
        # Conditional flow-matching objective L_FM (Sec. IV-E):
        # regress the velocity (x_1 - x_0) along the linear interpolant path.
        t = torch.rand(x_1.shape[0], device=x_1.device).type_as(x_1)
        x_0 = torch.randn_like(x_1) # Sample from base distribution (Gaussian)

        # Linear interpolation path p_t(x|x_1)
        x_t = (1 - t.unsqueeze(-1)) * x_0 + t.unsqueeze(-1) * x_1
        u_t = x_1 - x_0 # Target vector field

        v_t = flow_net(t, x_t, cond) # Predicted vector field

        return F.mse_loss(v_t, u_t)

    def _ode_solve(self, flow_net, z, cond, steps=10):
        # Simple Euler integrator of the learned flow ODE (used by the audit).
        h = 1.0 / steps
        x = z
        for i in range(steps):
            t = torch.tensor([i * h], device=z.device).repeat(z.shape[0])
            v = flow_net(t, x, cond)
            x = x + h * v
        return x

    def forward(self, v_feat, t_feat, cond_feat):
        # Flow-matching audit loss for each modality (Sec. IV-E)
        loss_fm_v = self._flow_matching_loss(self.v_flow_net, v_feat, cond_feat)
        loss_fm_t = self._flow_matching_loss(self.t_flow_net, t_feat, cond_feat)

        total_fm_loss = loss_fm_v + loss_fm_t

        # Cross-modal synergy: couples the visual and textual flows so the
        # between-modality gap G_u is well-defined (condition-separation reg.).
        s = torch.rand(1, device=v_feat.device) # Sample a time step s
        z_v = torch.randn_like(v_feat)
        z_t = torch.randn_like(t_feat)

        phi_s_v = self._ode_solve(self.v_flow_net, z_v, cond_feat, steps=int(s.item()*10)+1)
        phi_s_t = self._ode_solve(self.t_flow_net, z_t, cond_feat, steps=int(s.item()*10)+1)

        loss_cross = F.mse_loss(phi_s_v, self.text_to_visual_proj(phi_s_t))

        total_loss = total_fm_loss + self.lambda_cross * loss_cross
        return total_loss


    def _dispersion_trace(self, flow_net, x_data, cond):
        # Within-item dispersion proxy: the trace of the flow Jacobian
        # (the divergence Tr(d v / d x)) at a sampled path point. This is the
        # covariance-trace readout the served sensor reports as w_i (Sec. IV-B).

        t = torch.rand(x_data.shape[0], device=x_data.device).type_as(x_data)
        x_noise = torch.randn_like(x_data)

        # Form the point on the path at which to evaluate the trace
        x_t = ((1 - t.unsqueeze(-1)) * x_noise + t.unsqueeze(-1) * x_data).requires_grad_(True)

        v_t = flow_net(t, x_t, cond)

        # Trace Tr(d v_t / d x_t): sum the diagonal of the Jacobian by computing,
        # for each component i, the gradient of v_i w.r.t. x and reading entry i.
        dispersion = torch.zeros(x_t.shape[0], device=x_t.device)
        for i in range(x_t.shape[1]):
            grad_outputs = torch.zeros_like(v_t)
            grad_outputs[:, i] = 1
            # autograd.grad returns the VJP; with a one-hot seed this is a Jacobian row.
            j_row_i = torch.autograd.grad(outputs=v_t, inputs=x_t, grad_outputs=grad_outputs, retain_graph=True, create_graph=False)[0]
            # The i-th entry of that row is the diagonal element.
            dispersion += j_row_i[:, i]

        return dispersion.detach()

    def served_dispersion(self, v_feat, t_feat, cond_feat):
        # Per-modality within-item dispersion (w^v_i, w^t_i), read in a single
        # forward pass with no per-query ODE solve (Sec. IV-B).
        with torch.no_grad():
            disp_v = self._dispersion_trace(self.v_flow_net, v_feat, cond_feat)
            disp_t = self._dispersion_trace(self.t_flow_net, t_feat, cond_feat)

        return torch.abs(disp_v), torch.abs(disp_t)


class ResidualQuantizer(nn.Module):
    r"""A single residual-quantization codebook level of the intent hierarchy
    (Sec. IV-F). Assigns the residual to its nearest code and returns the
    quantized vector with the commitment/codebook losses; a straight-through
    estimator carries gradients past the discrete lookup."""

    def __init__(self, dim, codebook_size, commitment_weight=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_weight = commitment_weight
        self.codebook = nn.Embedding(codebook_size, dim)

    def forward(self, x):
        # x: (B, D)
        B, D = x.shape

        # Find nearest codebook entry
        x_flat = x.view(B, -1, D)
        distances = torch.sum(x_flat**2, dim=-1, keepdim=True) - \
                    2 * torch.matmul(x_flat, self.codebook.weight.t()) + \
                    torch.sum(self.codebook.weight**2, dim=-1, keepdim=False)

        indices = torch.argmin(distances, dim=-1) # (B, 1)
        quantized = self.codebook(indices).view(B, D)

        # Commitment term L_commit (Sec. IV-F) and codebook update term
        commitment_loss = F.mse_loss(x, quantized.detach())
        codebook_loss = F.mse_loss(quantized, x.detach())

        loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        return quantized, indices, loss


class AdaptiveIntentHierarchy(nn.Module):
    r"""Resolvability-typed adaptive intent depth over a residual-quantized
    hierarchy (Sec. IV-F and IV-G).

    The user seed ``h_u`` is decomposed into a coarse-to-fine sequence of intent
    codes by ``L`` (=``max_depth``) residual-quantization levels. A per-user
    halting rule reads the within-item fuzziness ``w_u`` and stops going deeper
    once the residual norm falls below a fuzziness-typed threshold: broad/clear
    users keep refining, fuzzy users halt early (the resolvability hypothesis,
    breadth -> deeper, fuzziness -> shallower)."""

    def __init__(self, dim, max_depth=4, codebook_size=128, halt_rate=1.0, lambda_div=0.1, diversity_margin=0.1):
        super().__init__()
        self.dim = dim
        self.max_depth = max_depth          # L in the paper
        self.halt_rate = halt_rate          # rate of the typed halting threshold
        self.lambda_div = lambda_div
        self.diversity_margin = diversity_margin

        self.quantizers = nn.ModuleList([
            ResidualQuantizer(dim, codebook_size) for _ in range(max_depth)
        ])

    def forward(self, h_u, w_u):
        # h_u: (B, D) user seed; w_u: (B,) within-item fuzziness W_u
        B, D = h_u.shape
        residual = h_u

        intent_codes = []
        total_commit_loss = 0.0

        for l in range(self.max_depth):
            # Resolvability-typed halting: deeper levels demand a smaller residual,
            # and fuzzier users (larger w_u) raise the threshold so they exit earlier.
            threshold = (self.halt_rate / (l + 1)) * w_u
            residual_norm = torch.norm(residual, p=2, dim=-1)

            # Mask of users who should keep refining at this depth
            active_mask = (residual_norm >= threshold).float().unsqueeze(-1)
            if active_mask.sum() == 0:
                break # Stop once no user is active

            # Quantize the residual for the still-active users
            quantizer = self.quantizers[l]
            quantized_level, _, commit_loss_level = quantizer(residual * active_mask)

            intent_codes.append(quantized_level)
            total_commit_loss += commit_loss_level

            # Update the residual only for active users
            residual = residual - (quantized_level * active_mask)


        # Codebook diversity regularizer (discourages code collapse)
        total_div_loss = 0.0
        if self.lambda_div > 0 and len(intent_codes) > 0:
            for l in range(self.max_depth):
                codebook = self.quantizers[l].codebook.weight
                c_dist = torch.cdist(codebook, codebook, p=2)
                loss_div_level = F.relu(self.diversity_margin - c_dist).mean()
                total_div_loss += loss_div_level

        commit_loss = total_commit_loss + self.lambda_div * total_div_loss
        return intent_codes, commit_loss




class DepthAttentionAggregator(nn.Module):
    r"""Depth-attention aggregation of the active intent codes (Sec. IV-G).

    Builds the depth-d user representation u_u^{(d)} = h_u + sum_l alpha_l c_u^{(l)}
    with content-based attention weights alpha_l (scaled by sqrt(dim)). A
    fuzziness-typed penalty down-weights deeper codes for fuzzy users, so depth
    is only consumed where the user's breadth is resolvable."""

    def __init__(self, dim, gamma=1.0):
        super().__init__()
        self.dim = dim
        self.gamma = gamma

    def forward(self, intent_codes, h_u, w_u):
        # intent_codes: list of (B, D) tensors; h_u: (B, D); w_u: (B,) fuzziness
        if not intent_codes:
            return torch.zeros_like(h_u)

        D_u = len(intent_codes)
        intents_stack = torch.stack(intent_codes, dim=1) # (B, D_u, D)

        # Content-based attention score <h_u, c_u^{(l)}> / sqrt(dim)
        scores = torch.einsum('bd,bld->bl', h_u, intents_stack) / math.sqrt(self.dim)

        # Fuzziness-typed depth penalty (larger for fuzzy users / deeper levels)
        levels = torch.arange(1, D_u + 1, device=h_u.device).float() # (D_u,)
        # Add a small epsilon to avoid division by zero
        penalty = self.gamma * levels / (w_u.unsqueeze(-1) + 1e-8) # (B, D_u)

        final_scores = scores - penalty
        attn_weights = F.softmax(final_scores, dim=1) # (B, D_u)

        # Aggregate intent codes by the attention weights
        aggregated_intent = torch.einsum('bl,bld->bd', attn_weights, intents_stack)
        return aggregated_intent



class TRIAD(GeneralRecommender):
    def __init__(self, config, dataset):
        super(TRIAD, self).__init__(config, dataset)

        # --- Hyperparameters for the TRIAD-specific modules ---
        self.lambda_cross = config['lambda_cross']  # between-modality synergy weight
        self.beta1 = config['beta1'] # weight of the dispersion (flow audit) loss
        self.beta2 = config['beta2'] # weight of the intent commitment loss


        device = self.device
        dim_x = config['embedding_size']
        # Within-item dispersion sensor (flow audit, Sec. IV-B / IV-E)
        self.flow_sensor = FlowDispersionSensor(
            feat_dim=dim_x,
            cond_dim=dim_x,
            hidden_dim=128,
            lambda_cross=self.lambda_cross
        ).to(device)

        # Resolvability-typed adaptive intent hierarchy (Sec. IV-F / IV-G)
        self.intent_hierarchy = AdaptiveIntentHierarchy(
            dim=dim_x,
            max_depth=config.get('intent_depth', 4),
            codebook_size=config.get('intent_codebook_size', 128),
            halt_rate=config.get('halt_rate', 1.0),
            lambda_div=config.get('lambda_div', 0.1)
        ).to(device)

        # Depth-attention aggregator producing u_u^{(d)} (Sec. IV-G)
        self.depth_aggregator = DepthAttentionAggregator(
            dim=dim_x,
            gamma=config.get('depth_gamma', 1.0)
        ).to(device)


        # ---- Multimodal graph backbone (Sec. IV-D) ----
        num_user = self.n_users
        num_item = self.n_items
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']

        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.epsilon = config['epsilon']
        self.lambda1 = config['lambda1']
        self.dim_latent = dim_x

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        # Load visual and textual features
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # Build item-item modality similarity graphs (frozen kNN, Sec. IV-D)
        if self.v_feat is not None:
            _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            self.v_mm_adj = image_adj
        if self.t_feat is not None:
            _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            self.t_mm_adj = text_adj
        if self.v_feat is not None and self.t_feat is not None:
            self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            del text_adj, image_adj

        # Build user-item interaction graph
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # Build user-user graph
        _, self.uu_adj = self.get_knn_uu_mat(self.edge_index)

        # Learnable layer-readout weights
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        # ID embedding
        self.id_feat = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32,
                                                requires_grad=True), gain=1).to(self.device))

        # Instantiate the backbone graph-conv modules (Sec. IV-D)
        if self.v_feat is not None:
            self.v_mlp = ModalityProjection(dim_latent=dim_x, features=self.v_feat)
            self.v_gcn = CollaborativeGraphConv(self.dataset, config['train_batch_size'], num_user, num_item, dim_x, 'add', dim_latent=dim_x, device=self.device, features=self.v_feat)
            self.ori_v_gcn = ContentGraphConv(self.dataset, config['train_batch_size'], num_user, num_item, dim_x, 'add', dim_latent=dim_x, device=self.device, features=self.v_feat)

        if self.t_feat is not None:
            self.t_mlp = ModalityProjection(dim_latent=dim_x, features=self.t_feat)
            self.t_gcn = CollaborativeGraphConv(self.dataset, config['train_batch_size'], num_user, num_item, dim_x, 'add', dim_latent=dim_x, device=self.device, features=self.t_feat)
            self.ori_t_gcn = ContentGraphConv(self.dataset, config['train_batch_size'], num_user, num_item, dim_x, 'add', dim_latent=dim_x, device=self.device, features=self.t_feat)

        self.id_gcn = CollaborativeGraphConv(self.dataset, config['train_batch_size'], num_user, num_item, dim_x, 'add',
                                             dim_latent=dim_x, device=self.device, features=self.id_feat)

        # A map from user index to their interacted items' indices
        self.user_item_map = {u:[] for u in range(num_user)}
        for u,i in zip(train_interactions.row, train_interactions.col):
            self.user_item_map[u].append(i)


    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1).expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices)


    def compute_normalized_laplacian(self, indices):
        adj_size = (indices.max().item() + 1, indices.max().item() + 1)
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)


    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))


    def _user_dispersion(self, users, item_dispersion):
        # Aggregate per-item dispersion over a user's consumed items to obtain
        # the per-user within-item fuzziness W_u (Sec. IV-C).
        w_u = torch.zeros(len(users), device=self.device)
        for i, user_id in enumerate(users.tolist()):
            interacted_items = self.user_item_map.get(user_id, [])
            if interacted_items:
                user_item_dispersion = item_dispersion[interacted_items]
                w_u[i] = user_item_dispersion.mean()
        return w_u + 1e-6 # Add epsilon for stability


    def calculate_all_embeddings(self):

        # 1. Backbone graph propagation for every modality (Sec. IV-D)
        self.vv_feat = self.v_mlp(self.v_feat)
        self.tt_feat = self.t_mlp(self.t_feat)

        self.vv_feat_gcn = self.ori_v_gcn(self.edge_index, self.v_mm_adj.coalesce().indices(), self.vv_feat)
        self.tt_feat_gcn = self.ori_t_gcn(self.edge_index, self.t_mm_adj.coalesce().indices(), self.tt_feat)

        self.v_rep, self.v_preference = self.v_gcn(self.edge_index, self.edge_index, self.vv_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index, self.edge_index, self.tt_feat)
        self.id_rep, self.id_preference = self.id_gcn(self.edge_index, self.edge_index, self.id_feat)

        all_users = torch.arange(self.n_users).to(self.device)
        all_items = torch.arange(self.n_items).to(self.device)

        # 2. Flow audit: estimate the within-item dispersion w_i (Sec. IV-B / IV-E)
        dispersion_loss = self.flow_sensor(
            self.vv_feat_gcn, self.tt_feat_gcn, self.id_rep[self.num_user:]
        )
        disp_v, disp_t = self.flow_sensor.served_dispersion(
            self.vv_feat_gcn, self.tt_feat_gcn, self.id_rep[self.num_user:]
        )
        item_dispersion = disp_v + disp_t

        # 3. Adaptive intent hierarchy: residual codes for every user (Sec. IV-F/G)
        user_dispersion = self._user_dispersion(all_users, item_dispersion)
        user_collab_rep = self.id_rep[:self.num_user]

        intent_codes, commit_loss = self.intent_hierarchy(user_collab_rep, user_dispersion)

        # 4. Depth-attention aggregation into u_u^{(d)} (Sec. IV-G)
        aggregated_intent = self.depth_aggregator(intent_codes, user_collab_rep, user_dispersion)

        # 5. Final user representation  u_u = h_u + sum_l alpha_l c_u^{(l)}
        user_final_rep = user_collab_rep + aggregated_intent

        # 6. Final item representation (collaborative + content branches)
        item_base_rep = torch.cat((self.v_rep[self.num_user:], self.t_rep[self.num_user:]), dim=1)
        i2i_graph_rep = self.buildItemGraph(self.mm_adj, item_base_rep)
        item_final_rep = item_base_rep + i2i_graph_rep

        # 7. Propagate through the U-I and U-U graphs
        u2u_graph_rep = self.buildItemGraph(self.uu_adj, user_final_rep)
        user_final_rep = user_final_rep + u2u_graph_rep

        result_embed = torch.cat((user_final_rep, item_final_rep), dim=0)
        final_embeddings = self.lightgcn_propagate(self.compute_normalized_laplacian(self.edge_index), result_embed)

        # Store for loss calculation
        self.dispersion_loss = dispersion_loss
        self.commit_loss = commit_loss

        return final_embeddings



    def forward(self, interaction):

        # The main forward pass computes all embeddings and then slices the batch
        self.result_embed1 = self.calculate_all_embeddings()

        # Two perturbed views for the contrastive loss
        self.perturbed_embeddings1 = self.lightgcn_propagate(self.compute_normalized_laplacian(self.edge_index), self.result_embed1, perturbed=True)
        self.perturbed_embeddings2 = self.lightgcn_propagate(self.compute_normalized_laplacian(self.edge_index), self.result_embed1, perturbed=True)

        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        self.u_idx = user_nodes
        self.v_idx = pos_item_nodes

        user_tensor = self.result_embed1[user_nodes]
        pos_item_tensor = self.result_embed1[pos_item_nodes + self.n_users]
        neg_item_tensor = self.result_embed1[neg_item_nodes + self.n_users]

        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)

        return pos_scores, neg_scores


    def buildItemGraph(self, adj, h):
        for i in range(self.n_layers):
            h = torch.sparse.mm(adj, h)
        return h


    def lightgcn_propagate(self, adj, all_embeddings, perturbed=False):
        # Layer-weighted LightGCN read-out over the user-item graph (Sec. IV-D)
        embeddings_list = [all_embeddings]
        if adj.dtype != all_embeddings.dtype:
            adj = adj.to(all_embeddings.dtype)

        for i in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj, all_embeddings)
            if perturbed:
                random_noise = torch.rand_like(all_embeddings)
                all_embeddings += torch.sign(all_embeddings) * F.normalize(random_noise, p=2, dim=-1) * self.epsilon
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        return lightgcn_all_embeddings


    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)

        # BPR ranking loss L_rank
        loss_rec = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

        # Embedding regularization
        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)

        # Self-supervised contrastive loss
        cl_loss = self.calc_cl_loss(self.perturbed_embeddings1, self.perturbed_embeddings2)

        # Full objective: rank + reg + cl + beta1 * dispersion(NLL/flow) + beta2 * commit
        total_loss = loss_rec + reg_loss + cl_loss + \
                     self.beta1 * self.dispersion_loss + \
                     self.beta2 * self.commit_loss

        return total_loss


    def calc_cl_loss(self, perturbed_embeddings1, perturbed_embeddings2):
        # Self-supervised contrastive loss over the two perturbed views
        unique_u_idx = torch.unique(self.u_idx)
        unique_v_idx = torch.unique(self.v_idx)
        p_user_emb1 = perturbed_embeddings1[unique_u_idx]
        p_user_emb2 = perturbed_embeddings2[unique_u_idx]
        p_item_emb1 = perturbed_embeddings1[unique_v_idx + self.n_users]
        p_item_emb2 = perturbed_embeddings2[unique_v_idx + self.n_users]

        normalize_emb_user1 = F.normalize(p_user_emb1, p=2, dim=1)
        normalize_emb_user2 = F.normalize(p_user_emb2, p=2, dim=1)
        normalize_emb_item1 = F.normalize(p_item_emb1, p=2, dim=1)
        normalize_emb_item2 = F.normalize(p_item_emb2, p=2, dim=1)

        pos_score_u = torch.exp((normalize_emb_user1 * normalize_emb_user2).sum(dim=-1) / 0.2)
        ttl_score_u = torch.exp(torch.matmul(normalize_emb_user1, normalize_emb_user2.t()) / 0.2).sum(dim=1)

        pos_score_i = torch.exp((normalize_emb_item1 * normalize_emb_item2).sum(dim=-1) / 0.2)
        ttl_score_i = torch.exp(torch.matmul(normalize_emb_item1, normalize_emb_item2.t()) / 0.2).sum(dim=1)

        cl_loss = - (torch.log(pos_score_u / ttl_score_u).mean() + torch.log(pos_score_i / ttl_score_i).mean())
        return self.lambda1 * cl_loss



    def full_sort_predict(self, interaction):
        # Scoring for evaluation:  y_ui = (u_u^*)^T h_i
        final_embeddings = self.calculate_all_embeddings()
        user_tensor = final_embeddings[:self.n_users]
        item_tensor = final_embeddings[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def get_knn_uu_mat(self, edge_index):
        """ We've omitted some non-core code and will release the full version promptly upon paper acceptance. """
        return None



class GraphConvLayer(MessagePassing):
    r"""LightGCN-style normalized neighbor-aggregation layer (Sec. IV-D)."""

    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(GraphConvLayer, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        """ We've omitted some non-core code and will release the full version promptly upon paper acceptance. """
        return None

    def message(self, x_j, edge_index, size):
        """ We've omitted some non-core code and will release the full version promptly upon paper acceptance. """
        return None


    def update(self, aggr_out):
        """ We've omitted some non-core code and will release the full version promptly upon paper acceptance. """
        return None





class ModalityProjection(torch.nn.Module):
    r"""Per-modality projection W_m (the 2-layer MLP that maps a raw, l2-normalized
    modality feature into the shared embedding space, Sec. IV-D)."""

    def __init__(self, dim_latent, features=None):
        super(ModalityProjection, self).__init__()
        self.dim_latent = dim_latent
        self.dim_feat = features.size(1)
        self.mlp_1 = nn.Linear(self.dim_feat, 4 * self.dim_latent)
        self.mlp_2 = nn.Linear(4 * self.dim_latent, self.dim_latent)

    def forward(self, features):
        features = self.mlp_1(features)
        features = F.leaky_relu(features)
        features = self.mlp_2(features)
        return features





class ContentGraphConv(torch.nn.Module):
    r"""Content branch: smooths the projected modality features over the frozen
    item-item kNN graph (Sec. IV-D)."""

    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode,
                 dim_latent=None, device=None, features=None):
        super(ContentGraphConv, self).__init__()
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.conv_embed_1 = GraphConvLayer(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index_drop, edge_index, features, perturbed=False):
        temp_features = features
        x = temp_features
        x = F.normalize(x)
        h = self.conv_embed_1(x, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h)
            h += torch.sign(h) * F.normalize(random_noise, dim=-1) * 0.1
        h_1 = self.conv_embed_1(h, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h_1)
            h_1 += torch.sign(h_1) * F.normalize(random_noise, dim=-1) * 0.1
        x_hat = x + h + h_1
        return x_hat



class CollaborativeGraphConv(torch.nn.Module):
    r"""Collaborative branch: LightGCN over learnable user/item ID embeddings on
    the user-item graph, with a learnable user preference table (Sec. IV-D)."""

    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode,
                 dim_latent=None, device=None, features=None):
        super(CollaborativeGraphConv, self).__init__()
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
            gain=1).to(device))
        self.conv_embed_1 = GraphConvLayer(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
        self.device = device


    def forward(self, edge_index_drop, edge_index, features, perturbed=False):
        x = torch.cat((self.preference, features), dim=0).to(self.device)
        x = F.normalize(x)
        h = self.conv_embed_1(x, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h)
            h += torch.sign(h) * F.normalize(random_noise, dim=-1) * 0.1
        h_1 = self.conv_embed_1(h, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h_1)
            h_1 += torch.sign(h_1) * F.normalize(random_noise, dim=-1) * 0.1
        x_hat = x + h + h_1
        return x_hat, self.preference
