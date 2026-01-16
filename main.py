
import time
import argparse
import pandas as pd
from tqdm import tqdm
from os.path import join
import os

from recognition.fov import fov_recon
from recognition.phase import phase_recon

tqdm.pandas()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to identify unknown FOV and phase in MRI or CT images.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file containing images' paths. Column with paths should be named 'im_path'.")
    parser.add_argument("im_type", type=str, choices=["MRI", "CT"], help="Images type in the indicated CSV.")
    parser.add_argument("--output", default=None, help="Path to save TotalSegmentation masks as Nifti. Masks are not saved by default.")
    parser.add_argument("--no_fov", action="store_true", help="Flag to indicate if FOV recognition should be run.")
    parser.add_argument("--no_phase", action="store_true", help="Flag to indicate if phase recognition should be run.")
    args = parser.parse_args()

    st = time.time()

    output_csv = args.input_file.replace('.csv', '_fov_phase_processed.csv')
    
    # 1. Load the original work to do
    df_input = pd.read_csv(args.input_file)
    assert 'im_path' in df_input.columns, "Input CSV must contain 'im_path' column."

    # 2. Filter out already processed paths
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        # Get list of paths already finished
        processed_paths = set(df_existing['im_path'].unique())
        # Keep only rows whose path is NOT in the processed list
        df_to_process = df_input[~df_input['im_path'].isin(processed_paths)].copy()
        print(f"Resuming: {len(processed_paths)} paths already processed. {len(df_to_process)} remaining.")
    else:
        df_to_process = df_input.copy()
        print(f"Starting fresh: {len(df_to_process)} paths to process.")

    if args.output is None:
        args.output = './tmp_masks/'
    os.makedirs(args.output, exist_ok=True)

    batch_size = 50 

    # 3. Process only the remaining rows
    for i in range(0, len(df_to_process), batch_size):
        batch = df_to_process.iloc[i:i + batch_size].copy()
        
        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} rows)")

        # FOV Recognition (Only if not already computed in THIS run)
        if not args.no_fov:
            # Note: We don't need the fov_mask here because we already filtered the whole DF
            batch['FOV'] = batch.progress_apply(
                lambda row: fov_recon(row["im_path"], args.im_type, join(args.output, str(row.name))), 
                axis=1
            )
                
        # Phase Recognition
        if not args.no_phase:
            batch['Phase'] = batch.progress_apply(
                lambda row: phase_recon(row["im_path"], args.im_type, join(args.output, str(row.name))), 
                axis=1
            )
        
        # 4. Append mode: Write only the new results to the end of the file
        # If it's the very first time the file is created, write header. Otherwise, don't.
        file_exists = os.path.isfile(output_csv)
        batch.to_csv(output_csv, mode='a', index=False, header=not file_exists)
    
    print(f"\nProcessing completed in {time.time() - st:.2f} seconds.")