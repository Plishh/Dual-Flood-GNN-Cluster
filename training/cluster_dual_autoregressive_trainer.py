import torch
import random
from typing import Tuple
from torch_geometric.utils import k_hop_subgraph
from .dual_autoregressive_trainer import DualAutoregressiveTrainer
from utils import train_utils

class ClusterDualAutoregressiveTrainer(DualAutoregressiveTrainer):
    """
    Cluster-based autoregressive trainer that:
    1. Uses standard DataLoader to yield full graph batches
    2. Randomly samples K clusters per batch
    3. Slices subgraphs for each cluster
    4. Runs autoregressive rollout on each cluster subgraph
    """
    def __init__(self, partition_map, clusters_per_batch=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if partition_map is None:
            raise ValueError("partition_map is required for ClusterDualAutoregressiveTrainer")
        self.partition_map = partition_map.to(self.device)
        self.clusters_per_batch = clusters_per_batch
        self.num_clusters = int(self.partition_map.max().item() + 1)
        self.training_stats.log(f"Cluster training: {self.num_clusters} clusters, sampling {clusters_per_batch} per batch")

    def _train_model(self, epoch: int, current_num_timesteps: int) -> Tuple[float, float, float, float, float]:
        """Performs autoregressive training on randomly sampled spatial clusters."""
        self.model.train()

        running_pred_loss = 0.0
        running_edge_pred_loss = 0.0
        running_global_mass_loss = 0.0
        running_local_mass_loss = 0.0

        total_cluster_steps = 0

        for full_batch in self.dataloader:
            full_batch = full_batch.to(self.device)
            
            # Randomly select clusters to process for this batch
            selected_clusters = random.sample(range(self.num_clusters), 
                                             min(self.clusters_per_batch, self.num_clusters))
            
            for cluster_id in selected_clusters:
                self.optimizer.zero_grad()

                # 1. Identify nodes in the current cluster
                node_mask = (self.partition_map == cluster_id)
                node_indices = node_mask.nonzero(as_tuple=True)[0]
                
                # 2. Slice the subgraph using k_hop_subgraph (handles edge re-indexing)
                subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                    node_indices, 
                    num_hops=0, 
                    edge_index=full_batch.edge_index, 
                    relabel_nodes=True
                )
                
                # 3. Create cluster-specific batch
                cluster_batch = full_batch.clone()
                cluster_batch.x = full_batch.x[subset]
                cluster_batch.edge_attr = full_batch.edge_attr[edge_mask]
                cluster_batch.edge_index = sub_edge_index
                cluster_batch.y = full_batch.y[subset]
                cluster_batch.y_edge = full_batch.y_edge[edge_mask]
                
                # Handle boundary masks if they exist (must be sliced and remapped to cluster indices)
                if hasattr(full_batch, 'boundary_nodes_mask'):
                    # Slice mask to cluster nodes only
                    cluster_batch.boundary_nodes_mask = full_batch.boundary_nodes_mask[subset].clone()
                if hasattr(full_batch, 'boundary_edges_mask'):
                    # Slice mask to cluster edges only
                    cluster_batch.boundary_edges_mask = full_batch.boundary_edges_mask[edge_mask].clone()

                # 4. Run autoregressive loop on this cluster (same as parent trainer logic)
                total_batch_loss = 0.0
                total_batch_pred_loss = 0.0
                total_batch_edge_pred_loss = 0.0
                total_batch_global_mass_loss = 0.0
                total_batch_local_mass_loss = 0.0

                x, edge_attr = cluster_batch.x[:, :, 0], cluster_batch.edge_attr[:, :, 0]
                edge_index = cluster_batch.edge_index

                sliding_window = x[:, self.start_node_target_idx:self.end_node_target_idx].clone()
                edge_sliding_window = edge_attr[:, self.start_edge_target_idx:self.end_edge_target_idx].clone()
                
                for i in range(current_num_timesteps):
                    x, edge_attr = cluster_batch.x[:, :, i], cluster_batch.edge_attr[:, :, i]

                    # Override with sliding window
                    x = torch.concat([x[:, :self.start_node_target_idx], sliding_window, x[:, self.end_node_target_idx:]], dim=1)
                    edge_attr = torch.concat([edge_attr[:, :self.start_edge_target_idx], edge_sliding_window, edge_attr[:, self.end_edge_target_idx:]], dim=1)

                    pred_diff, edge_pred_diff = self.model(x, edge_index, edge_attr)
                    pred_diff, edge_pred_diff = self._override_pred_bc(pred_diff, edge_pred_diff, cluster_batch, i)

                    pred_loss = self._compute_node_loss(pred_diff, cluster_batch, i)
                    pred_loss = self._scale_node_pred_loss(epoch, pred_loss)
                    total_batch_pred_loss += pred_loss.item()

                    edge_pred_loss = self._compute_edge_loss(edge_pred_diff, cluster_batch, i)
                    edge_pred_loss = self._scale_edge_pred_loss(epoch, pred_loss, edge_pred_loss)
                    total_batch_edge_pred_loss += edge_pred_loss.item()

                    step_loss = pred_loss + edge_pred_loss

                    prev_node_pred = sliding_window[:, [-1]]
                    pred = prev_node_pred + pred_diff
                    prev_edge_pred = edge_sliding_window[:, [-1]]
                    edge_pred = prev_edge_pred + edge_pred_diff

                    if self.use_physics_loss:
                        global_loss, local_loss = self._get_physics_loss(epoch, pred, prev_node_pred,
                                                                         prev_edge_pred, pred_loss, cluster_batch,
                                                                         current_timestep=i)
                        total_batch_global_mass_loss += global_loss.item()
                        total_batch_local_mass_loss += local_loss.item()
                        step_loss = step_loss + global_loss + local_loss

                    total_batch_loss = total_batch_loss + step_loss

                    if i < current_num_timesteps - 1:
                        next_sliding_window = torch.cat((sliding_window[:, 1:], pred), dim=1)
                        next_edge_sliding_window = torch.cat((edge_sliding_window[:, 1:], edge_pred), dim=1)
                        sliding_window = next_sliding_window
                        edge_sliding_window = next_edge_sliding_window

                # 5. Backward and optimizer step for this cluster
                avg_batch_loss = total_batch_loss / current_num_timesteps
                avg_batch_loss.backward()
                self._clip_gradients()
                self.optimizer.step()

                # Accumulate running losses
                total_losses = (total_batch_pred_loss, total_batch_edge_pred_loss, 
                               total_batch_global_mass_loss, total_batch_local_mass_loss)
                avg_losses = train_utils.divide_losses(total_losses, current_num_timesteps)
                avg_pred_loss, avg_edge_pred_loss, avg_global_mass_loss, avg_local_mass_loss = avg_losses
                
                running_pred_loss += avg_pred_loss
                running_edge_pred_loss += avg_edge_pred_loss
                running_global_mass_loss += avg_global_mass_loss
                running_local_mass_loss += avg_local_mass_loss
                
                total_cluster_steps += 1

        # Average results across all processed clusters
        running_loss = running_pred_loss + running_edge_pred_loss + running_global_mass_loss + running_local_mass_loss
        running_losses = (running_loss, running_pred_loss, running_edge_pred_loss, 
                         running_global_mass_loss, running_local_mass_loss)
        epoch_losses = train_utils.divide_losses(running_losses, total_cluster_steps)
        epoch_loss, pred_epoch_loss, edge_pred_epoch_loss, global_mass_epoch_loss, local_mass_epoch_loss = epoch_losses

        return epoch_loss, pred_epoch_loss, edge_pred_epoch_loss, global_mass_epoch_loss, local_mass_epoch_loss
