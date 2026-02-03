
import os
import csv, _csv
import time
import argparse
import pandas as pd
from tqdm import tqdm
from os.path import join

from recognition.fov import fov_recon
from recognition.phase import phase_recon
from recognition.utils import read_nifti, check_setting

tqdm.pandas()

def process_image(image_path: str, im_type: str, output_path: str | None, writer: _csv.Writer, nofov_nophase: tuple[bool, bool] = (False, False)):
    """Process a single image to determine its FOV and Phase."""
    assert len(nofov_nophase) == 2, "nofov_nophase must be a tuple of two boolean values."
    
    out = [image_path]
    img_obj, img_data = read_nifti(image_path, to_ras=True) # Load image once

    # FOV Recognition
    if not nofov_nophase[0]:
        try:
            fov = fov_recon(img_obj, img_data, im_type, output_path)
        except Exception as e:
            print(f"Error processing FOV for image {image_path}: {e}")
            fov = "error"
        out.append(fov)
    
    # Phase Recognition
    if not nofov_nophase[1]:
        try:
            phase = phase_recon(img_obj, im_type, output_path)
        except Exception as e:
            print(f"Error processing Phase for image {image_path}: {e}")
            phase = "error"
        out.append(phase)
    
    writer.writerow(out)

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
    
    # Load the work to do
    df_input = pd.read_csv(args.input_file)
    assert 'im_path' in df_input.columns, "Input CSV must contain 'im_path' column."

    # Filter out already processed images
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        check_setting(df_existing.columns, args.no_fov, args.no_phase)
        processed_paths = set(df_existing['im_path'].unique())
        df_to_process = df_input[~df_input['im_path'].isin(processed_paths)].copy()
        print(f"Resuming: {len(processed_paths)} paths already processed. {len(df_to_process)} remaining.")
    else:
        df_to_process = df_input.copy()
        print(f"Starting fresh: {len(df_to_process)} paths to process.")


    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if 'processed_paths' not in locals(): # write header if new file
            header = ['im_path']
            if not args.no_fov:
                header.append('fov')
            if not args.no_phase:
                header.append('phase')
            writer.writerow(header)
        df_to_process.progress_apply(lambda row: (process_image(row["im_path"],
                                                            args.im_type,
                                                            join(args.output, str(row.name))
                                                              if args.output is not None else None,
                                                            writer,
                                                            (args.no_fov, args.no_phase)
                                                            ), csvfile.flush()),
                                    axis=1
                                    )
    
    print(f"\nProcessing completed in {time.time() - st:.2f} seconds.")