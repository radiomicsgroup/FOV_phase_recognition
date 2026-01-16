
import time
import argparse
import pandas as pd
from tqdm import tqdm
from os.path import join

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

    df = pd.read_csv(args.input_file)
    assert 'im_path' in df.columns, "Input CSV must contain 'im_path' column."

    if args.output is None:
        args.output = './tmp_masks/'
    if not args.no_fov:
        df['FOV'] = df.progress_apply(lambda row: fov_recon(row["im_path"], args.im_type, join(args.output, str(row.name))), axis=1)
    if not args.no_phase:
        df['Phase'] = df.progress_apply(lambda row: phase_recon(row["im_path"], args.im_type, join(args.output, str(row.name))), axis=1)
    
    df.to_csv(args.input_file.replace('.csv', '_fov_phase_processed.csv'), index=False)
    
    print(f"Processing completed in {time.time() - st:.2f} seconds.")
