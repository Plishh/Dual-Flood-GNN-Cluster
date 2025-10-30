import numpy as np
import os
import traceback
import torch
import gc
import random
import torch.optim as optim # <<< CLUSTER-GCN MODIFICATION >>> (Ensure optim is imported)

from argparse import ArgumentParser, Namespace
from datetime import datetime
from torch_geometric.loader import ClusterLoader, DataLoader # <<< CLUSTER-GCN MODIFICATION >>> (Import ClusterLoader)
from torch_geometric.data import Data # <<< CLUSTER-GCN MODIFICATION >>>

from data import dataset_factory, FloodEventDataset
from models import model_factory
from test import get_test_dataset_config, run_test
# <<< CLUSTER-GCN MODIFICATION >>> (Import partitioning utils and the cluster trainer)
try:
    from utils.cluster_utils import load_base_graph_structure, partition_graph
    from training import trainer_factory, ClusterDualAutoregressiveTrainer # Assuming your cluster trainer is here
except ImportError as e:
    print(f"Warning: Could not import Cluster-GCN specific components: {e}")
    # Define dummy functions/classes if needed for script to load
    def load_base_graph_structure(*args, **kwargs): raise NotImplementedError("Import failed")
    def partition_graph(*args, **kwargs): raise NotImplementedError("Import failed")
    class ClusterDualAutoregressiveTrainer: pass # Dummy class

from training import trainer_factory # Keep original factory for non-cluster case
from typing import Dict, Optional, Tuple
from utils import Logger, file_utils, train_utils

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for training')
    parser.add_argument("--with_test", type=bool, default=False, help='Whether to run test after training')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    # <<< CLUSTER-GCN MODIFICATION >>> (Add arg for enabling cluster GCN)
    parser.add_argument("--use_cluster_gcn", action='store_true', help='Enable Cluster-GCN training strategy')
    return parser.parse_args()

# --- load_dataset function remains largely the same, prepares dataset objects ---
# --- It might need adjustment if validation also needs ClusterLoader ---
def load_dataset(config: Dict, args: Namespace, logger: Logger, use_cluster_gcn: bool) -> Tuple[FloodEventDataset, Optional[FloodEventDataset]]:
    dataset_parameters = config['dataset_parameters']
    root_dir = dataset_parameters['root_dir']
    train_dataset_parameters = dataset_parameters['training']
    loss_func_parameters = config['loss_func_parameters']
    base_datset_config = {
        'root_dir': root_dir,
        'nodes_shp_file': dataset_parameters['nodes_shp_file'],
        'edges_shp_file': dataset_parameters['edges_shp_file'],
        'features_stats_file': dataset_parameters['features_stats_file'],
        'previous_timesteps': dataset_parameters['previous_timesteps'],
        'normalize': dataset_parameters['normalize'],
        'timestep_interval': dataset_parameters['timestep_interval'],
        'spin_up_time': dataset_parameters['spin_up_time'],
        'time_from_peak': dataset_parameters['time_from_peak'],
        'inflow_boundary_nodes': dataset_parameters['inflow_boundary_nodes'],
        'outflow_boundary_nodes': dataset_parameters['outflow_boundary_nodes'],
        'with_global_mass_loss': loss_func_parameters.get('use_global_mass_loss', False), # Use .get for safety
        'with_local_mass_loss': loss_func_parameters.get('use_local_mass_loss', False),
        'debug': args.debug,
        'logger': logger,
        'force_reload': False, # Usually False unless dataset params change
    }

    dataset_summary_file = train_dataset_parameters['dataset_summary_file']
    event_stats_file = train_dataset_parameters['event_stats_file']
    # <<< CLUSTER-GCN MODIFICATION >>> (Force memory mode if using Cluster GCN)
    storage_mode = 'memory' if use_cluster_gcn else dataset_parameters['storage_mode']
    if use_cluster_gcn and dataset_parameters['storage_mode'] != 'memory':
        logger.log("Warning: Forcing 'memory' storage mode for Cluster-GCN compatibility.")

    train_config = config['training_parameters']
    early_stopping_patience = train_config['early_stopping_patience']

    # --- Determine if autoregressive training is enabled ---
    autoregressive_train_params = train_config.get('autoregressive', {})
    autoregressive_enabled = autoregressive_train_params.get('enabled', False)
    num_label_timesteps = autoregressive_train_params.get('total_num_timesteps', 1) # Default to 1 if not specified

    # --- Handle Validation Split ---
    val_dataset = None
    if early_stopping_patience is not None:
        percent_validation = train_config['val_split_percent']
        assert percent_validation is not None, 'Validation split percentage must be specified if early stopping is used.'
        logger.log(f'Splitting dataset events with {percent_validation * 100}% for validation')
        train_summary_file, val_summary_file = train_utils.split_dataset_events(root_dir, dataset_summary_file, percent_validation)

        # Config for Validation Dataset (usually not autoregressive for validation step)
        val_event_stats_file = val_summary_file.replace(os.path.basename(dataset_summary_file), event_stats_file) # Use basename
        val_dataset_config = {
            'mode': 'test', # Use test mode for validation loading logic
            'dataset_summary_file': val_summary_file,
            'event_stats_file': val_event_stats_file,
            **base_datset_config,
            # Validation typically doesn't need num_label_timesteps unless tester uses it
        }
        logger.log(f'Using validation dataset configuration: {val_dataset_config}')
        # <<< CLUSTER-GCN MODIFICATION >>> (Validation dataset also needs memory mode if using Cluster GCN validation)
        val_storage_mode = 'memory' if use_cluster_gcn else dataset_parameters.get('validation_storage_mode', storage_mode)
        val_dataset = dataset_factory(val_storage_mode, autoregressive=False, **val_dataset_config) # Val dataset usually not AR itself

    else:
        # No validation split needed
        train_summary_file = dataset_summary_file


    # --- Config for Training Dataset ---
    #train_event_stats_file = os.path.basename(train_summary_file).replace(os.path.basename(dataset_summary_file), event_stats_file) # Use basename
    train_event_stats_file = train_summary_file.replace(dataset_summary_file, event_stats_file)

    train_dataset_config = {
        'mode': 'train',
        'dataset_summary_file': train_summary_file,
        'event_stats_file': train_event_stats_file,
        **base_datset_config,
    }
    if autoregressive_enabled:
        train_dataset_config['num_label_timesteps'] = num_label_timesteps # Add AR specific arg

    logger.log(f'Using training dataset configuration: {train_dataset_config}')
    train_dataset = dataset_factory(storage_mode, autoregressive=autoregressive_enabled, **train_dataset_config)

    logger.log(f'Loaded train dataset with {len(train_dataset)} samples.')
    if val_dataset:
        logger.log(f'Loaded validation dataset with {len(val_dataset)} samples.')

    return train_dataset, val_dataset


# --- run_train function remains largely the same, prepares and calls trainer ---
def run_train(model: torch.nn.Module,
              model_name: str,
              # <<< CLUSTER-GCN MODIFICATION >>> (Accept loader instead of dataset)
              train_loader: torch.utils.data.DataLoader, # Accepts ClusterLoader or DataLoader
              logger: Logger,
              config: Dict,
              val_dataset: Optional[FloodEventDataset] = None, # Keep val_dataset for trainer validation logic
              stats_dir: Optional[str] = None,
              model_dir: Optional[str] = None,
              device: str = 'cpu',
              use_cluster_gcn: bool = False) -> str: # <<< CLUSTER-GCN MODIFICATION >>>
        train_config = config['training_parameters']

        # Loss function and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['adam_weight_decay'])
        logger.log(f'Using Adam optimizer with learning rate {train_config["learning_rate"]} and weight decay {train_config["adam_weight_decay"]}')

        base_trainer_params = train_utils.get_trainer_config(model_name, config, logger)
        trainer_params = {
            'model': model,
            'dataloader': train_loader, # <<< CLUSTER-GCN MODIFICATION >>> (Pass the loader)
            'val_dataset': val_dataset,
            'optimizer': optimizer,
            'logger': logger,
            'device': device,
            **base_trainer_params,
        }

        autoregressive_train_config = train_config.get('autoregressive', {})
        autoregressive_enabled = autoregressive_train_config.get('enabled', False)

        # <<< CLUSTER-GCN MODIFICATION >>> (Select the correct trainer class)
        if use_cluster_gcn:
            # Ensure the specific trainer class is imported and available
            try:
                # Assuming ClusterDualAutoregressiveTrainer exists in training package
                from training import ClusterDualAutoregressiveTrainer
                logger.log("Using ClusterDualAutoregressiveTrainer.")
                trainer = ClusterDualAutoregressiveTrainer(**trainer_params)
            except ImportError:
                 raise ImportError("ClusterDualAutoregressiveTrainer not found. Make sure it's defined and imported.")
            except Exception as e:
                 raise RuntimeError(f"Error initializing ClusterDualAutoregressiveTrainer: {e}")
        else:
             logger.log("Using standard trainer factory.")
             trainer = trainer_factory(model_name, autoregressive_enabled, **trainer_params)
        # <<< END CLUSTER-GCN MODIFICATION >>>

        trainer.train() # Call the train method of the selected trainer

        trainer.print_stats_summary()

        # Save training stats and model
        curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if stats_dir is not None:
            if not os.path.exists(stats_dir):
                os.makedirs(stats_dir)

            saved_metrics_path = os.path.join(stats_dir, f'{model_name}_{curr_date_str}_train_stats.npz')
            trainer.save_stats(saved_metrics_path)

        model_path = f'{model_name}_{curr_date_str}.pt'
        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_path = os.path.join(model_dir, f'{model_name}_{curr_date_str}.pt')
            trainer.save_model(model_path)

        return model_path

def main():
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)

    train_config = config['training_parameters']
    log_path = train_config['log_path']
    logger = Logger(log_path=log_path)

    try:
        logger.log('================================================')

        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                 torch.cuda.manual_seed_all(args.seed)
            logger.log(f'Setting random seed to {args.seed}')

        current_device = torch.cuda.get_device_name(args.device) if args.device == 'cuda' and torch.cuda.is_available() else 'CPU'
        logger.log(f'Using device: {current_device}')

        # <<< CLUSTER-GCN MODIFICATION >>> (Determine if using Cluster GCN early)
        use_cluster_gcn = args.use_cluster_gcn
        if use_cluster_gcn:
            logger.log("Cluster-GCN training mode enabled.")
            # Add Cluster-GCN specific configs (can also be moved to config file)
            cluster_gcn_config = config.get('cluster_gcn_parameters', {}) # Get from config if exists
            NUM_CLUSTERS = cluster_gcn_config.get('num_clusters', 100) # Default 100
            BATCH_SIZE_CLUSTERS = cluster_gcn_config.get('clusters_per_batch', 20) # Default 20 clusters per batch
            logger.log(f"Cluster-GCN Params: Num Clusters={NUM_CLUSTERS}, Clusters per Batch={BATCH_SIZE_CLUSTERS}")


        # --- Dataset Loading ---
        logger.log("Loading dataset...")
        train_dataset, val_dataset = load_dataset(config, args, logger, use_cluster_gcn)

        # --- Cluster-GCN Preprocessing (if enabled) ---
        if use_cluster_gcn:
            logger.log("Performing Cluster-GCN preprocessing...")
            dataset_parameters = config['dataset_parameters']
            root_dir = dataset_parameters['root_dir']
            raw_dir = os.path.join(root_dir, 'raw')
            nodes_shp = dataset_parameters['nodes_shp_file']
            edges_shp = dataset_parameters['edges_shp_file']
            # Find a representative HEC-RAS file for geometry
            # Using the first one from the training summary list
            train_summary_path = os.path.join(raw_dir, train_dataset_parameters['dataset_summary_file'])
            if not os.path.exists(train_summary_path):
                 # Fallback if split_dataset_events wasn't run or file missing
                 train_summary_path = os.path.join(raw_dir, dataset_parameters['training']['dataset_summary_file'])

            try:
                import pandas as pd
                summary_df = pd.read_csv(train_summary_path)
                hec_ras_geo_file = summary_df['HECRAS_Filepath'][0] # Use first file's geometry
            except Exception as e:
                logger.log(f"Error reading HECRAS_Filepath from {train_summary_path}: {e}. Falling back to config.")
                # Fallback: Define a default HECRAS geo file in config if needed
                hec_ras_geo_file = dataset_parameters.get('hecras_geometry_file', 'default_geo.hdf') # Get from config or define default

            # 1. Load static structure
            logger.log("Loading base graph structure for partitioning...")
            base_graph_structure: Data = load_base_graph_structure(
                raw_dir, nodes_shp, edges_shp, hec_ras_geo_file
            )

            # 2. Partition the graph
            logger.log("Partitioning graph with METIS...")
            clustered_structure = partition_graph(base_graph_structure, NUM_CLUSTERS)
            partition_data = clustered_structure.part

            # 3. Add partition info to the loaded dataset
            # Ensure the train_dataset is InMemoryDataset type
            if not hasattr(train_dataset, 'data'):
                 raise TypeError("Cluster-GCN requires an InMemory dataset with a 'data' attribute.")
            logger.log("Adding partition information to the loaded dataset...")
            if train_dataset.data.num_nodes != clustered_structure.num_nodes:
                raise ValueError(
                    f"Node count mismatch! Partitioning used {clustered_structure.num_nodes} "
                    f"but loaded dataset has {train_dataset.data.num_nodes}. "
                    "Ensure the same base graph is used for partitioning and data loading."
                )
            train_dataset.data.part = partition_data
            logger.log("Partition information added.")

            # 4. Create ClusterLoader
            logger.log("Setting up ClusterLoader...")
            train_loader = ClusterLoader(
                train_dataset.data, # Pass the Data object with 'part' attribute
                num_workers=0,
                batch_size=BATCH_SIZE_CLUSTERS,
                shuffle=True,
            )
        else:
            # --- Standard DataLoader ---
            logger.log("Setting up standard DataLoader...")
            # Use batch size from config file
            batch_size = train_config.get('batch_size', 64) # Default batch size
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0 # Adjust as needed
            )
        # <<< END CLUSTER-GCN MODIFICATION >>>


        # --- Model Initialization ---
        logger.log("Initializing model...")
        model_params = config['model_parameters'][args.model]
        # Calculate input sizes dynamically based on the loaded dataset object
        # Ensure dataset object has these properties after loading
        base_model_params = {
            'static_node_features': train_dataset.num_static_node_features,
            'dynamic_node_features': train_dataset.num_dynamic_node_features,
            'static_edge_features': train_dataset.num_static_edge_features,
            'dynamic_edge_features': train_dataset.num_dynamic_edge_features,
            'previous_timesteps': train_dataset.previous_timesteps,
            'device': args.device,
        }
        model_config = {**model_params, **base_model_params}
        model = model_factory(args.model, **model_config)
        logger.log(f'Using model: {args.model}')
        logger.log(f'Using model configuration: {model_config}')
        num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # Get num params
        logger.log(f'Number of trainable model parameters: {num_train_params}')

        checkpoint_path = train_config.get('checkpoint_path', None)
        if checkpoint_path is not None:
            logger.log(f'Loading model from checkpoint: {checkpoint_path}')
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location=args.device)) # Removed weights_only for broader compatibility
            except Exception as e:
                logger.log(f"Error loading checkpoint: {e}. Check compatibility.")
                # Decide whether to proceed with random weights or exit

        model.to(args.device) # Ensure model is on the correct device

        # --- Run Training ---
        logger.log("Starting training run...")
        stats_dir = train_config['stats_dir']
        model_dir = train_config['model_dir']
        model_path = run_train(model=model,
                               model_name=args.model,
                               # <<< CLUSTER-GCN MODIFICATION >>> (Pass loader)
                               train_loader=train_loader,
                               val_dataset=val_dataset,
                               logger=logger,
                               config=config,
                               stats_dir=stats_dir,
                               model_dir=model_dir,
                               device=args.device,
                               # <<< CLUSTER-GCN MODIFICATION >>> (Pass flag)
                               use_cluster_gcn=use_cluster_gcn)

        logger.log('================================================')
        logger.log(f'Training finished. Saved model to: {model_path}')

        # --- Optional Testing ---
        if not args.with_test:
            return

        # =================== Testing ===================
        logger.log(f'\nStarting testing for model: {model_path}')

        # Test dataset setup (remains the same, testing usually done on full graph or specific events)
        dataset_parameters = config['dataset_parameters']
        base_datset_config = {
            'root_dir': dataset_parameters['root_dir'],
            'nodes_shp_file': dataset_parameters['nodes_shp_file'],
            'edges_shp_file': dataset_parameters['edges_shp_file'],
            'features_stats_file': dataset_parameters['features_stats_file'],
            'previous_timesteps': dataset_parameters['previous_timesteps'],
            'normalize': dataset_parameters['normalize'],
            'timestep_interval': dataset_parameters['timestep_interval'],
            'spin_up_time': dataset_parameters['spin_up_time'],
            'time_from_peak': dataset_parameters['time_from_peak'],
            'inflow_boundary_nodes': dataset_parameters['inflow_boundary_nodes'],
            'outflow_boundary_nodes': dataset_parameters['outflow_boundary_nodes'],
            'debug': args.debug,
            'logger': logger,
            'force_reload': False, # Avoid reprocessing test data unless needed
        }
        test_dataset_config = get_test_dataset_config(base_datset_config, config)
        logger.log(f'Using test dataset configuration: {test_dataset_config}')

        # Clear memory before loading test dataset
        del train_dataset
        del val_dataset
        del train_loader
        # <<< CLUSTER-GCN MODIFICATION >>> (Clear cluster specific vars if needed)
        if use_cluster_gcn:
             del base_graph_structure, clustered_structure, partition_data
        gc.collect()
        if args.device == 'cuda': torch.cuda.empty_cache()

        storage_mode = dataset_parameters.get('test_storage_mode', 'disk') # Allow separate test storage mode
        test_dataset = dataset_factory(storage_mode, autoregressive=False, **test_dataset_config)
        logger.log(f'Loaded test dataset with {len(test_dataset)} samples')

        # Load the trained model state again for testing
        # model.load_state_dict(torch.load(model_path, map_location=args.device)) # Reload best model
        # Or just use the model object directly if run_train returns the trained model state

        logger.log(f'Using model checkpoint for {args.model}: {model_path}')
        logger.log(f'Using model configuration: {model_config}')

        test_config = config['testing_parameters']
        rollout_start = test_config['rollout_start']
        rollout_timesteps = test_config['rollout_timesteps']
        output_dir = test_config['output_dir']
        run_test(model=model, # Pass the trained model object
                 model_path=model_path, # Path for logging purposes
                 dataset=test_dataset,
                 logger=logger,
                 rollout_start=rollout_start,
                 rollout_timesteps=rollout_timesteps,
                 output_dir=output_dir,
                 device=args.device)

        logger.log('================================================')

    except Exception as e:
        logger.log(f'Unexpected error in main:\n{traceback.format_exc()}') # Log full traceback


if __name__ == '__main__':
    # Ensure imports for data_utils, model, training, etc. work
    # Add necessary paths if files are in different directories
    # e.g., sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main()

