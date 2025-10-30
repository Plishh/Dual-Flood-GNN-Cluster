import numpy as np
import os
import argparse

# --- Configuration ---
# These filenames are based on your dataset scripts.
CONSTANT_VALUES_FILE = 'constant_values.npz'
BOUNDARY_MASK_FILE = 'boundary_condition_masks.npz'

# Index of 'area' in the static_nodes array, based on:
# STATIC_NODE_FEATURES = ['area', 'roughness', 'elevation']
AREA_FEATURE_INDEX = 0

def calculate_stats(root_dir: str):
    """
    Calculates and prints graph statistics from processed dataset files.

    Args:
        root_dir: The root directory of your dataset (containing
                  'processed' and 'raw' subfolders).
    """
    processed_dir = os.path.join(root_dir, 'processed')

    # --- File Paths ---
    constants_path = os.path.join(processed_dir, CONSTANT_VALUES_FILE)
    boundary_mask_path = os.path.join(processed_dir, BOUNDARY_MASK_FILE)

    # --- Check if files exist ---
    if not os.path.exists(constants_path):
        print(f"Error: Could not find processed file: {constants_path}")
        print("Please ensure you have run the dataset processing script first.")
        return

    if not os.path.exists(boundary_mask_path):
        print(f"Error: Could not find processed file: {boundary_mask_path}")
        print("Please ensure you have run the dataset processing script first.")
        return

    try:
        # --- Load Processed Data ---
        print(f"Loading data from {constants_path}...")
        constant_values = np.load(constants_path)
        static_nodes = constant_values['static_nodes']

        print(f"Loading data from {boundary_mask_path}...")
        boundary_masks = np.load(boundary_mask_path)
        # This mask is True for nodes that are part of the boundary
        boundary_nodes_mask = boundary_masks['boundary_nodes_mask']

        # --- 1. Count Total Number of Nodes ---
        # This is the total number of nodes in the final graph,
        # including real cells and boundary nodes.
        total_nodes = static_nodes.shape[0]
        
        # --- 2. Calculate Average Cell Area ---
        
        # Create a mask for "real" nodes (i.e., not boundary nodes)
        # As seen in boundary_condition.py, boundary nodes are added
        # with static features (like area) set to 0.
        # We must exclude these from the average area calculation.
        real_nodes_mask = ~boundary_nodes_mask
        
        num_real_nodes = np.sum(real_nodes_mask)
        num_boundary_nodes = total_nodes - num_real_nodes

        # Get the 'area' feature for real nodes only
        real_node_areas = static_nodes[real_nodes_mask, AREA_FEATURE_INDEX]
        
        # Calculate the average area
        average_area = real_node_areas.mean()

        # --- Print Results ---
        print("\n--- Graph Statistics ---")
        print(f"Total Nodes (Real Cells + Boundary Nodes): {total_nodes}")
        print(f"  - Real Cell Nodes:    {num_real_nodes}")
        print(f"  - Boundary Nodes:   {num_boundary_nodes}")
        print(f"\nAverage Area of a single Real Cell: {average_area:.2f} (units squared)")
        print("------------------------")

    except KeyError as e:
        print(f"Error: Missing expected data in .npz file: {e}")
        print("The processed files might be from an older or different version.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Set up a simple command-line argument parser
    parser = argparse.ArgumentParser(
        description="Calculate node count and average cell area from processed GNN dataset."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="The root directory of your dataset (e.g., './data/my_flood_dataset')."
    )
    
    args = parser.parse_args()
    
    calculate_stats(args.root_dir)
