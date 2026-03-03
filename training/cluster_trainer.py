import os
import time
import torch

from contextlib import redirect_stdout
from torch import Tensor
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph
from .dual_autoregressive_trainer import DualAutoregressiveTrainer
from typing import Tuple, Callable
from utils import EarlyStopping, LossScaler, train_utils, metric_utils, cluster_utils
import random

from .node_autoregressive_trainer import NodeAutoregressiveTrainer
from .edge_autoregressive_trainer import EdgeAutoregressiveTrainer

class ClusterTrainer(DualAutoregressiveTrainer):
    def __init__(self, partition_map, clusters_per_batch=3, sliding=True,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        if partition_map is None:
            raise ValueError("partition_map is required for ClusterDualAutoregressiveTrainer")
        self.partition_map = partition_map.to(self.device)
        self.clusters_per_batch = clusters_per_batch
        self.num_clusters = int(self.partition_map.max().item() + 1)
        self.training_stats.log(f"Cluster training: {self.num_clusters} clusters, sampling {clusters_per_batch} per batch, is sliding: {sliding}")
        self.sliding = sliding  


    def train(self):
        '''Multi-step-ahead loss with curriculum learning.'''
        self.training_stats.log("Cluster trainer starting training...")
        self.training_stats.start_train()
        current_num_timesteps = self.init_num_timesteps
        current_timestep_epochs = 0
        self.training_stats.log(f"device is {self.device}")
        for epoch in range(self.num_epochs):
            train_start_time = time.time()
            self.training_stats.log(
                f'\tCurriculum state: epoch={epoch + 1}, current_num_timesteps={current_num_timesteps}, '
                f'epochs_in_stage={current_timestep_epochs}'
            )

            train_losses = self._train_model(epoch, current_num_timesteps)
            epoch_loss, pred_epoch_loss, edge_pred_epoch_loss, global_mass_epoch_loss, local_mass_epoch_loss = train_losses

            logging_str = f'Epoch [{epoch + 1}/{self.num_epochs}]\n'
            logging_str += f'\tLoss: {epoch_loss:.4e}\n'
            logging_str += f'\tNode Prediction Loss: {pred_epoch_loss:.4e}\n'
            logging_str += f'\tEdge Prediction Loss: {edge_pred_epoch_loss:.4e}'
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(epoch_loss)
            self.training_stats.add_loss_component('prediction_loss', pred_epoch_loss)
            self.training_stats.add_loss_component('edge_prediction_loss', edge_pred_epoch_loss)

            if self.use_physics_loss:
                self._log_epoch_physics_loss(global_mass_epoch_loss, local_mass_epoch_loss)

            self._update_loss_scaler_for_epoch(epoch)

            train_end_time = time.time()
            train_duration = train_end_time - train_start_time
            self.training_stats.log(f'\tEpoch Train Duration: {train_duration:.2f} seconds')

            val_node_rmse, val_edge_rmse = self.validate()
            self.training_stats.log(f'\n\tValidation Node RMSE: {val_node_rmse:.4e}')
            self.training_stats.log(f'\tValidation Edge RMSE: {val_edge_rmse:.4e}')
            self.training_stats.add_val_loss_component('val_node_rmse', val_node_rmse)
            self.training_stats.add_val_loss_component('val_edge_rmse', val_edge_rmse)

            current_timestep_epochs += 1

            is_early_stopped = self.early_stopping((val_node_rmse, val_edge_rmse), self.model)
            is_max_exceeded = self.max_curriculum_epochs is not None and current_timestep_epochs >= self.max_curriculum_epochs
            if is_early_stopped or is_max_exceeded:
                if current_num_timesteps < self.total_num_timesteps:
                    self.training_stats.log(f'\tCurriculum learning for {current_num_timesteps} steps ended after {current_timestep_epochs} epochs.')
                    current_num_timesteps += self.timestep_increment
                    current_timestep_epochs = 0
                    self.early_stopping = EarlyStopping(patience=self.early_stopping.patience)
                    self.training_stats.log(f'\tIncreased current_num_timesteps to {current_num_timesteps} timesteps.')
                    self.lr_scheduler.step()
                    self.training_stats.log(f'\tDecayed learning rate to {self.lr_scheduler.get_last_lr()[0]:.4e}.')
                    continue

                self.training_stats.log('Training completed due to early stopping.')
                break

        self.training_stats.end_train()
        self.training_stats.add_additional_info('edge_scaled_loss_ratios', self.edge_loss_scaler.scaled_loss_ratio_history)
        self._add_scaled_physics_loss_history()

    def _train_model(self, epoch: int, current_num_timesteps: int) -> Tuple[float, float, float, float, float]:
        """Performs autoregressive training on SPATIALLY AGGREGATED clusters."""
        self.model.train()

        if self.device == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

        running_pred_loss = 0.0
        running_edge_pred_loss = 0.0
        running_global_mass_loss = 0.0
        running_local_mass_loss = 0.0

        total_cluster_steps = 0
        total_full_batches = 0
        total_cluster_groups = 0
        total_clusters_per_group = 0
        total_subgraph_nodes = 0
        total_subgraph_edges = 0
        self.training_stats.log("Starting new full batch iteration...")
        self.training_stats.log(f"{len(self.dataloader)} {len(self.dataset)} {self.batch_size}")
        for full_batch in self.dataloader:
            total_full_batches += 1
            full_batch = full_batch.to(self.device)

            # PREPARATION: Identify Batch Structure 
            if hasattr(full_batch, 'batch'):
                graph_ids = full_batch.batch.unique().tolist()
                # Use the first graph to define the spatial 'template' for clustering
                # This assumes all graphs in the batch share the same underlying topology/partition map
                mask_first_graph = (full_batch.batch == graph_ids[0])
                # Filter edges that belong only to the first graph for the template
                edge_mask_0 = mask_first_graph[full_batch.edge_index[0]] & mask_first_graph[full_batch.edge_index[1]]
                template_edge_index = full_batch.edge_index[:, edge_mask_0]
            else:
                graph_ids = [0]
                template_edge_index = full_batch.edge_index

            # Determine Spatial Clusters 
            # We decide which clusters to train on ONCE for the whole batch based on the template
            
            if self.sliding:
                cluster_groups = cluster_utils.get_centered_neighbor_groups(
                    template_edge_index, self.partition_map
                )
            else:
                cluster_groups = cluster_utils.get_clusters_list(
                    self.num_clusters, self.clusters_per_batch
                )

            if len(cluster_groups) == 0:
                continue

            total_cluster_groups += len(cluster_groups)
            total_clusters_per_group += sum(len(group) for group in cluster_groups)

            # Iterate over Spatial Locations 
            for selected_clusters in cluster_groups:
                
                self.optimizer.zero_grad()

                # Aggregate Nodes from ALL Graphs 
                all_batch_node_indices = []
                
                # We need to map the "Local Cluster ID" to "Global Node IDs" for every graph in the batch
                selected_clusters_tensor = torch.tensor(selected_clusters, device=self.partition_map.device)
                local_mask = torch.isin(self.partition_map, selected_clusters_tensor)

                for graph_id in graph_ids:
                    if hasattr(full_batch, 'batch'):
                        # Find where this specific graph lives in the global array
                        node_indices_global = (full_batch.batch == graph_id).nonzero(as_tuple=True)[0]
                        if node_indices_global.numel() == 0: continue
                    else:
                        node_indices_global = torch.arange(full_batch.num_nodes, device=self.device)

                    # Map: Local Mask -> Global Indices for this specific graph
                    # This selects the specific cluster nodes for *this* graph instance
                    global_nodes_for_graph = node_indices_global[local_mask]
                    all_batch_node_indices.append(global_nodes_for_graph)

                if not all_batch_node_indices:
                    continue
                
                # Combine into one massive list of indices to slice at once
                combined_node_indices = torch.cat(all_batch_node_indices)

                # --- STEP 4: Create ONE Combined Subgraph ---
                # This is the most efficient way: PyG handles re-indexing for the whole batch
                subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                    combined_node_indices,
                    num_hops=0,
                    edge_index=full_batch.edge_index,
                    relabel_nodes=True
                )

                total_subgraph_nodes += int(subset.numel())
                total_subgraph_edges += int(edge_mask.sum().item())

                # 3. Create combined cluster-specific batch
                cluster_batch = full_batch.clone()
                cluster_batch.x = full_batch.x[subset]
                cluster_batch.edge_attr = full_batch.edge_attr[edge_mask]
                cluster_batch.edge_index = sub_edge_index
                cluster_batch.y = full_batch.y[subset]
                cluster_batch.y_edge = full_batch.y_edge[edge_mask]

                # Update PyG batch index tensor to match cluster size
                if hasattr(full_batch, 'batch'):
                    cluster_batch.batch = full_batch.batch[subset].clone()
                    # Re-index to 0 since we only have 1 graph per combined cluster
                    cluster_batch.batch = torch.zeros_like(cluster_batch.batch)

                # Update num_nodes to match cluster size
                cluster_batch.num_nodes = subset.size(0)

                # Handle boundary masks if they exist (must be sliced and remapped to cluster indices)
                if hasattr(full_batch, 'boundary_nodes_mask'):
                    cluster_batch.boundary_nodes_mask = full_batch.boundary_nodes_mask[subset].clone()
                if hasattr(full_batch, 'boundary_edges_mask'):
                    cluster_batch.boundary_edges_mask = full_batch.boundary_edges_mask[edge_mask].clone()

                # Update physics loss info dictionaries with cluster-specific masks
                # Copy all fields but slice node/edge masks to match cluster size
                if self.use_global_loss and hasattr(full_batch, 'global_mass_info'):
                    cluster_batch.global_mass_info = full_batch.global_mass_info.copy()
                    cluster_batch.global_mass_info['non_boundary_nodes_mask'] = full_batch.global_mass_info['non_boundary_nodes_mask'][subset].clone()
                    cluster_batch.global_mass_info['inflow_edges_mask'] = full_batch.global_mass_info['inflow_edges_mask'][edge_mask].clone()
                    cluster_batch.global_mass_info['outflow_edges_mask'] = full_batch.global_mass_info['outflow_edges_mask'][edge_mask].clone()
                    # Aggregate rainfall from all graphs since cluster nodes are now treated as batch 0
                    cluster_batch.global_mass_info['total_rainfall'] = full_batch.global_mass_info['total_rainfall'].sum(dim=0, keepdim=True)

                if self.use_local_loss and hasattr(full_batch, 'local_mass_info'):
                    cluster_batch.local_mass_info = full_batch.local_mass_info.copy()
                    cluster_batch.local_mass_info['non_boundary_nodes_mask'] = full_batch.local_mass_info['non_boundary_nodes_mask'][subset].clone()
                    cluster_batch.local_mass_info['rainfall'] = full_batch.local_mass_info['rainfall'][subset].clone()
                # 4. Run autoregressive loop on the combined cluster batch
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

                # 5. Backward and optimizer step for the combined cluster batch
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

        avg_groups_per_full_batch = total_cluster_groups / max(total_full_batches, 1)
        avg_clusters_per_group = total_clusters_per_group / max(total_cluster_groups, 1)
        avg_subgraph_nodes = total_subgraph_nodes / max(total_cluster_steps, 1)
        avg_subgraph_edges = total_subgraph_edges / max(total_cluster_steps, 1)
        self.training_stats.log(
            f'\tCluster workload (epoch): full_batches={total_full_batches}, '
            f'optimizer_steps={total_cluster_steps}, groups={total_cluster_groups}, '
            f'avg_groups_per_batch={avg_groups_per_full_batch:.2f}, '
            f'avg_clusters_per_group={avg_clusters_per_group:.2f}, '
            f'avg_subgraph_nodes={avg_subgraph_nodes:.1f}, '
            f'avg_subgraph_edges={avg_subgraph_edges:.1f}'
        )
        if self.device == 'cuda':
            peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            self.training_stats.log(f'\tCUDA peak memory allocated: {peak_mem_gb:.2f} GB')

        # Final Epoch Stats
        running_loss = running_pred_loss + running_edge_pred_loss + running_global_mass_loss + running_local_mass_loss
        
        # Helper to average over all processed clusters
        epoch_losses = train_utils.divide_losses(
            (running_loss, running_pred_loss, running_edge_pred_loss, running_global_mass_loss, running_local_mass_loss), 
            total_cluster_steps
        )

        return epoch_losses
    
    def validate(self):
        self.model.eval()

        event_node_loss_list, event_edge_loss_list, event_global_loss_list, event_local_loss_list = [], [], [], []
        event_node_rmse_list, event_edge_rmse_list = [], []

        epoch = self.num_epochs_dyn_loss + 1
        non_boundary_nodes_mask = ~self.boundary_nodes_mask
        ds = self.val_dataset
        for event_idx in range(len(ds.hec_ras_run_ids)):
            with torch.no_grad():
                event_start_idx = ds.event_start_idx[event_idx]
                event_end_idx = ds.event_start_idx[event_idx + 1] if event_idx + 1 < len(ds.event_start_idx) else ds.total_rollout_timesteps
                event_dataset = ds[event_start_idx:event_end_idx]
                dataloader = DataLoader(event_dataset, batch_size=1, shuffle=False) # Enforce batch size = 1 for autoregressive testing

                node_loss_list, edge_loss_list, global_loss_list, local_loss_list = [], [], [], []
                node_rmse_list, edge_rmse_list = [], []

                sliding_window = event_dataset[0].x[:, self.start_node_target_idx:self.end_node_target_idx].clone()
                edge_sliding_window = event_dataset[0].edge_attr[:, self.start_edge_target_idx:self.end_edge_target_idx].clone()
                sliding_window, edge_sliding_window = sliding_window.to(self.device), edge_sliding_window.to(self.device)
                for graph in dataloader:
                    # ========== Inference ==========
                    graph = graph.to(self.device)

                    x = torch.concat([graph.x[:, :self.start_node_target_idx], sliding_window, graph.x[:, self.end_node_target_idx:]], dim=1)
                    edge_attr = torch.concat([graph.edge_attr[:, :self.start_edge_target_idx], edge_sliding_window, graph.edge_attr[:, self.end_edge_target_idx:]], dim=1)
                    edge_index = graph.edge_index

                    pred_diff, edge_pred_diff = self.model(x, edge_index, edge_attr)

                    # Override boundary conditions in predictions
                    pred_diff[self.boundary_nodes_mask] = graph.y[self.boundary_nodes_mask]
                    edge_pred_diff[self.boundary_edges_mask] = graph.y_edge[self.boundary_edges_mask]

                    # ========== Training Losses ==========
                    pred_loss = self.loss_func(pred_diff, graph.y)
                    pred_loss = self._scale_node_pred_loss(epoch, pred_loss)
                    node_loss_list.append(pred_loss)

                    edge_pred_loss = self.edge_loss_func(edge_pred_diff, graph.y_edge)
                    edge_pred_loss = self._scale_edge_pred_loss(epoch, None, edge_pred_loss)
                    edge_loss_list.append(edge_pred_loss)

                    prev_node_pred = sliding_window[:, [-1]]
                    pred = prev_node_pred + pred_diff
                    prev_edge_pred = edge_sliding_window[:, [-1]]
                    edge_pred = prev_edge_pred + edge_pred_diff

                    if self.use_global_loss:
                        global_mass_loss = self._get_global_mass_loss(epoch,pred, prev_node_pred,
                                                                      prev_edge_pred, None, graph)
                        global_loss_list.append(global_mass_loss)
                    if self.use_local_loss:
                        local_mass_loss = self._get_local_mass_loss(epoch, pred, prev_node_pred,
                                                                    prev_edge_pred, None, graph)
                        local_loss_list.append(local_mass_loss)

                    sliding_window = torch.concat((sliding_window[:, 1:], pred), dim=1)
                    edge_sliding_window = torch.concat((edge_sliding_window[:, 1:], edge_pred), dim=1)

                    # ========== Validation Metrics ==========
                    label = graph.x[:, [self.end_node_target_idx-1]] + graph.y
                    if ds.is_normalized:
                        pred = ds.normalizer.denormalize(ds.NODE_TARGET_FEATURE, pred)
                        label = ds.normalizer.denormalize(ds.NODE_TARGET_FEATURE, label)

                    # Ensure water volume is non-negative
                    pred = torch.clip(pred, min=0)
                    label = torch.clip(label, min=0)

                    # Filter boundary conditions for metric computation
                    pred = pred[non_boundary_nodes_mask]
                    label = label[non_boundary_nodes_mask]

                    node_rmse = metric_utils.RMSE(pred.cpu(), label.cpu())
                    node_rmse_list.append(node_rmse)

                    label_edge = graph.edge_attr[:, [self.end_edge_target_idx-1]] + graph.y_edge
                    if ds.is_normalized:
                        edge_pred = ds.normalizer.denormalize(ds.EDGE_TARGET_FEATURE, edge_pred)
                        label_edge = ds.normalizer.denormalize(ds.EDGE_TARGET_FEATURE, label_edge)

                    edge_rmse = metric_utils.RMSE(edge_pred.cpu(), label_edge.cpu())
                    edge_rmse_list.append(edge_rmse)

                event_node_loss_list.append(torch.stack(node_loss_list).mean())
                event_edge_loss_list.append(torch.stack(edge_loss_list).mean())

                if self.use_global_loss:
                    event_global_loss_list.append(torch.stack(global_loss_list).mean())
                if self.use_local_loss:
                    event_local_loss_list.append(torch.stack(local_loss_list).mean())

                event_node_rmse_list.append(torch.stack(node_rmse_list).mean())
                event_edge_rmse_list.append(torch.stack(edge_rmse_list).mean())

        # Store training losses for validation
        avg_node_loss = torch.stack(event_node_loss_list).mean().item()
        avg_edge_loss = torch.stack(event_edge_loss_list).mean().item()
        self.training_stats.add_val_loss_component('val_node_loss', avg_node_loss)
        self.training_stats.add_val_loss_component('val_edge_loss', avg_edge_loss)

        if self.use_global_loss:
            avg_global_loss = torch.stack(event_global_loss_list).mean().item()
            self.training_stats.add_val_loss_component('val_global_mass_loss', avg_global_loss)

        if self.use_local_loss:
            avg_local_loss = torch.stack(event_local_loss_list).mean().item()
            self.training_stats.add_val_loss_component('val_local_mass_loss', avg_local_loss)

        node_rmse = torch.stack(event_node_rmse_list).mean().item()
        edge_rmse = torch.stack(event_edge_rmse_list).mean().item()

        return node_rmse, edge_rmse

    # def validate(self):
    #     val_tester = DualAutoregressiveTester(
    #         model=self.model,
    #         dataset=self.val_dataset,
    #         include_physics_loss=False,
    #         device=self.device
    #     )
    #     with open(os.devnull, "w") as f, redirect_stdout(f):
    #         val_tester.test()

    #     node_rmse = val_tester.get_avg_node_rmse()
    #     edge_rmse = val_tester.get_avg_edge_rmse()
    #     return node_rmse, edge_rmse

    def _compute_edge_loss(self, edge_pred: Tensor, batch, timestep: int) -> Tensor:
        label = batch.y_edge[:, :, timestep]
        return self.edge_loss_func(edge_pred, label)

    def _override_pred_bc(self, pred: Tensor, edge_pred: Tensor, batch, timestep: int) -> Tuple[Tensor, Tensor]:
        """Override boundary conditions using cluster-specific masks from batch."""
        # Use cluster-specific boundary masks from the batch object
        if hasattr(batch, 'boundary_nodes_mask'):
            batch_boundary_nodes_mask = batch.boundary_nodes_mask
            pred[batch_boundary_nodes_mask] = batch.y[batch_boundary_nodes_mask, :, timestep]
        
        if hasattr(batch, 'boundary_edges_mask'):
            batch_boundary_edges_mask = batch.boundary_edges_mask
            edge_pred[batch_boundary_edges_mask] = batch.y_edge[batch_boundary_edges_mask, :, timestep]
        
        return pred, edge_pred

# ========= Methods for scaling losses =========

    def _scale_edge_pred_loss(self, epoch: int, basis_loss: Tensor, edge_pred_loss: Tensor) -> Tensor:
        if epoch < self.num_epochs_dyn_loss:
            self.edge_loss_scaler.add_epoch_loss_ratio(basis_loss, edge_pred_loss)
            scaled_edge_pred_loss = self.edge_loss_scaler.scale_loss(edge_pred_loss)
        else:
            scaled_edge_pred_loss = self.edge_loss_scaler.scale_loss(edge_pred_loss) * self.edge_loss_weight
        return scaled_edge_pred_loss

    def _update_loss_scaler_for_epoch(self, epoch: int):
        if epoch < self.num_epochs_dyn_loss:
            self.edge_loss_scaler.update_scale_from_epoch()
            self.training_stats.log(f'\tAdjusted Edge Pred Loss Weight to {self.edge_loss_scaler.scale:.4e}')
        NodeAutoregressiveTrainer._update_loss_scaler_for_epoch(self, epoch)
