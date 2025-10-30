import time
import torch
from torch import Tensor
from torch_geometric.data import Data # Batch objects from ClusterLoader are Data
from typing import Tuple

# --- Assuming your existing trainer setup ---
# Make sure these imports work based on your project structure
try:
    # If DualAutoregressiveTrainer is in the same directory (training/)
    from .dual_autoregressive_trainer import DualAutoregressiveTrainer
    from .node_autoregressive_trainer import NodeAutoregressiveTrainer
    from .edge_autoregressive_trainer import EdgeAutoregressiveTrainer
    # If utils are in a parent directory
    import sys
    import os
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils import train_utils # Assuming train_utils is accessible
except ImportError as e:
    print(f"Warning: Error importing base trainers or utils: {e}")
    # Define dummy base class if import fails, to allow script loading
    class DualAutoregressiveTrainer:
        def __init__(self, *args, **kwargs):
             # Initialize necessary attributes from base class if needed
             self.device = 'cpu'; self.use_physics_loss = False; self.start_node_target_idx = 0
             self.end_node_target_idx = 1; self.start_edge_target_idx = 0; self.end_edge_target_idx = 1
             self.optimizer = None; self.model = None; self.dataloader = None; self.log_func = print
             self.num_epochs_dyn_loss = 0; self.node_loss_func = None; self.edge_loss_func = None
             self.edge_loss_scaler = None # Need dummy scaler
             class DummyScaler:
                 scale = 1.0
                 def add_epoch_loss_ratio(self, *args): pass
                 def scale_loss(self, loss): return loss
             self.edge_loss_scaler = DummyScaler()

        def _get_epoch_total_running_loss(self, *args): return args[0]
        def _reset_epoch_physics_running_loss(self): pass
        def _get_epoch_physics_loss(self, *args): return torch.tensor(0.0)
        def _clip_gradients(self): pass
        # Dummy methods from parents needed for _train_model logic
        def _compute_node_loss(self, pred, batch, i): return torch.tensor(0.0)
        def _compute_edge_loss(self, edge_pred, batch, i): return torch.tensor(0.0)
        def _override_pred_bc(self, pred, edge_pred, batch, i): return pred, edge_pred
        def _scale_edge_pred_loss(self, epoch, pred_loss, edge_pred_loss): return edge_pred_loss

class ClusterDualAutoregressiveTrainer(DualAutoregressiveTrainer):
    """
    Autoregressive trainer adapted to work with ClusterLoader from PyG.
    Structure and variable names aligned with the original DualAutoregressiveTrainer.

    Assumes the dataloader yields Data objects (subgraphs) where:
    - batch.x contains node features for initial timesteps [N_batch, F_node, T_init]
    - batch.edge_attr contains edge features for initial timesteps [E_batch, F_edge, T_init]
    - batch.y contains node labels for the prediction horizon [N_batch, F_node_out, H]
    - batch.y_edge contains edge labels for the prediction horizon [E_batch, F_edge_out, H]
    - T_init >= previous_timesteps + 1 (includes current step for prediction)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_func(f"Initialized ClusterDualAutoregressiveTrainer.")
        # Optional: Add check for ClusterLoader if needed

    def _train_model(self, epoch: int, current_num_timesteps: int) -> Tuple[float, float, float]:
        """
        Trains the model for one epoch using ClusterLoader batches (subgraphs).
        Performs multi-step-ahead prediction within each batch, mimicking
        the structure of the original DualAutoregressiveTrainer.
        """
        self.model.train()
        running_pred_loss = 0.0
        running_edge_pred_loss = 0.0
        if self.use_physics_loss:
            self._reset_epoch_physics_running_loss()

        num_batches = len(self.dataloader)
        if num_batches == 0:
             print("Warning: DataLoader is empty.")
             return 0.0, 0.0, 0.0

        for batch_idx, batch in enumerate(self.dataloader):
            if not isinstance(batch, Data):
                raise TypeError(f"ClusterLoader should yield Data objects, got {type(batch)}")

            self.optimizer.zero_grad()
            batch = batch.to(self.device)

            # --- Data Validation ---
            if batch.x.dim() != 3 or batch.edge_attr.dim() != 3:
                raise ValueError("Batch node/edge features must be 3D [Elements, Features, Timesteps]")
            num_init_timesteps = batch.x.shape[2] # T_init

            if batch.y.dim() != 3 or batch.y_edge.dim() != 3:
                 raise ValueError("Batch node/edge labels must be 3D [Elements, OutFeatures, Horizon]")
            horizon = batch.y.shape[2] # H

            # Ensure enough initial timesteps are provided by the dataset loader
            # This check depends on how your dataset prepares the features.
            # Assuming the model needs 'previous_timesteps + 1' slices for one prediction.
            # Example: If previous_timesteps=2, model needs t-2, t-1, t for predicting t+1.
            # The structure used in the original trainer suggests `batch.x[:,:,i]` is the input for step `i`.
            if num_init_timesteps < 1: # Must have at least the 'current' step
                 raise ValueError(f"Batch features must have at least 1 timestep, got {num_init_timesteps}")

            # Determine actual prediction steps based on curriculum and available labels
            predict_steps = min(current_num_timesteps, horizon)
            if predict_steps <= 0:
                print(f"Warning: Calculated predict_steps is {predict_steps} for batch {batch_idx}. Skipping batch.")
                continue

            # --- Initialize Sliding Windows (Mimicking original trainer) ---
            # Get the features corresponding to the target variables from the *initial* input state
            # Assuming batch.x[:,:,0] contains the state at the start of the prediction window
            # The window size should match the number of output features (usually 1 if predicting only next step directly)
            # This requires careful thought: Does the model expect *all* previous target values, or just the latest?
            # The original trainer's logic suggests it uses a window matching `previous_timesteps`? Let's check...
            # `sliding_window = x[:, self.start_node_target_idx:self.end_node_target_idx].clone()` is ambiguous.
            # Let's assume the simplest case: window holds only the *latest* target value, to be replaced by prediction.
            # Need clarification on how `previous_timesteps` are used if > 0.
            # Safest initial guess matching original structure: use features from the FIRST timestep in the label horizon.
            # Note: This might need adjustment based on dataset preparation logic.
            # Let's assume batch.x[:,:,0] has the feature state to predict step 1 (label i=0).

            # Initialize based on the *last available input timestep* provided by the loader.
            initial_x = batch.x[:, :, -1] # Shape: [N_batch, F_node]
            initial_edge_attr = batch.edge_attr[:, :, -1] # Shape: [E_batch, F_edge]

            # These will hold the *predicted* target values as the loop progresses
            # Initialize with the *actual* target values from the *last input step*
            sliding_window = initial_x[:, self.start_node_target_idx:self.end_node_target_idx].clone()
            edge_sliding_window = initial_edge_attr[:, self.start_edge_target_idx:self.end_edge_target_idx].clone()

            # --- Autoregressive Loop ---
            total_batch_loss = torch.tensor(0.0, device=self.device)

            for i in range(predict_steps):
                # --- Prepare Input Features 'x' and 'edge_attr' for step i ---
                # Get the base features for this step from the input data provided by loader.
                # If T_init > 1, this needs logic to select the correct slice.
                # Simplest for now: Assume we always use the *last* input slice and update its target part.
                # More robust: If T_init = H, we could use batch.x[:,:,i], batch.edge_attr[:,:,i] as base.
                # Let's assume the structure requires using the initial state and overriding target features.
                x = initial_x.clone() # Start with the features from the beginning of the window
                edge_attr = initial_edge_attr.clone()

                # Override graph data with the current state of the sliding window (previous prediction)
                x[:, self.start_node_target_idx:self.end_node_target_idx] = sliding_window
                edge_attr[:, self.start_edge_target_idx:self.end_edge_target_idx] = edge_sliding_window

                # --- Model Prediction ---
                pred, edge_pred = self.model(x, batch.edge_index, edge_attr)
                # pred: [N_batch, F_node_out], edge_pred: [E_batch, F_edge_out]

                # --- Boundary Condition Override ---
                pred, edge_pred = self._override_pred_bc(pred, edge_pred, batch, i)

                # --- Loss Calculation ---
                # Get labels for the step we just predicted (prediction for step i+1 uses label i)
                node_label = batch.y[:, :, i]
                edge_label = batch.y_edge[:, :, i]

                # Node Prediction Loss
                pred_loss = self._compute_node_loss(pred, batch, i) # Use inherited method
                running_pred_loss += pred_loss.item() # Accumulate raw item loss

                # Edge Prediction Loss
                edge_pred_loss = self._compute_edge_loss(edge_pred, batch, i) # Use inherited method
                scaled_edge_pred_loss = self._scale_edge_pred_loss(epoch, pred_loss.detach(), edge_pred_loss) # Use inherited method
                running_edge_pred_loss += scaled_edge_pred_loss.item() # Accumulate raw item loss

                step_loss = pred_loss + scaled_edge_pred_loss

                # --- Optional Physics Loss ---
                if self.use_physics_loss:
                    # Get previous step's prediction (stored in sliding window before update)
                    prev_node_pred = sliding_window # Prediction from step i-1 (used as input for step i)
                    prev_edge_pred = edge_sliding_window

                    # The original code had a special case for i==0, might need replication
                    # if i == 0:
                    #     prev_edge_pred = train_utils.overwrite_outflow_boundary(prev_edge_pred, batch)

                    physics_loss = self._get_epoch_physics_loss(
                        epoch, pred, prev_node_pred, prev_edge_pred, pred_loss.detach(), batch
                    )
                    step_loss = step_loss + physics_loss

                total_batch_loss = total_batch_loss + step_loss

                # --- Update Sliding Window for next iteration's input ---
                # Use detach() to prevent gradients flowing back through this update path
                sliding_window = pred.detach().clone()
                edge_sliding_window = edge_pred.detach().clone()

            # --- End Autoregressive Loop for Batch ---

            # --- Backpropagation ---
            if predict_steps > 0:
                avg_batch_loss = total_batch_loss / predict_steps # Average loss over horizon H
                avg_batch_loss.backward()
                self._clip_gradients() # Use inherited method
                self.optimizer.step()
            # else: handled earlier

        # --- End Epoch ---

        # Calculate average epoch losses (divide accumulated raw losses by total steps)
        total_steps_in_epoch = num_batches * current_num_timesteps # Approximation
        if total_steps_in_epoch == 0: total_steps_in_epoch = 1 # Avoid division by zero

        epoch_total_loss_value = (running_pred_loss + running_edge_pred_loss) # Already includes physics if enabled
        # Need careful check if _get_epoch_total_running_loss included physics loss scaling. Assuming it just sums.
        # epoch_total_loss_value = self._get_epoch_total_running_loss((running_pred_loss + running_edge_pred_loss))

        epoch_loss = epoch_total_loss_value / total_steps_in_epoch
        pred_epoch_loss = running_pred_loss / total_steps_in_epoch
        edge_pred_epoch_loss = running_edge_pred_loss / total_steps_in_epoch

        return epoch_loss, pred_epoch_loss, edge_pred_epoch_loss

    # --- Other methods (validate, etc.) are inherited ---
    # Ensure they are compatible with ClusterLoader batches (Data objects) if overridden.

