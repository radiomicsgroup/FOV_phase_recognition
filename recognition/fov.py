import os
import numpy as np
from typing import Literal
from os.path import join
from totalsegmentator.python_api import totalsegmentator

from recognition.utils import arr_union, read_nifti, remove_blank_slices

def read_mask(path: str, organ: str, non_blank = slice(None)) -> np.ndarray:
    """Read the segmentation mask for a specific organ."""
    if organ == "kidney":
        mask = arr_union([read_nifti(join(path,f"{part}.nii.gz"))[1]
                        for part in ["kidney_left", "kidney_right"]])
    else:
        _, mask = read_nifti(join(path,f"{organ}.nii.gz"))
    return mask[:,:,non_blank]

def fov_recon(image_path: str, im_type: Literal["MRI", "CT"], output_path: str) -> str:
    """
    Determine the anatomical field of view (FOV) of `image_path`.

    Arguments:
    - image_path (str): Path to the image file.
    - im_type (str): Type of the image ('MRI' or 'CT').
    - output_path (str): Path to save segmentation masks.

    Returns:
    - str: Recognized FOV value.
    """
    image_obj, image = read_nifti(image_path, to_ras=True)
    image, non_blank_slices = remove_blank_slices(image)
    if image.shape[2] < 10:
        return "scout"
    
    task = "total" if im_type == "CT" else "total_mr"
    organs = ["heart", "sternum", "sacrum", "liver", "spleen", "kidney"]
    if im_type == "MRI":
        organs.remove("sternum")  # sternum not available in total_mr model
    
    # totalsegmentator(image_obj, output_path, task=task)

    masks = {o: read_mask(output_path,o, non_blank_slices) for o in organs}

    heart_ok = masks["heart"].sum() > 0
    if im_type == "MRI":
        sternum_ok = True
        edges_clear = all(masks["heart"][:,:,i].sum() == 0 for i in (0, -1))
    else:
        sternum_ok = masks["sternum"].sum() > 0
        edges_clear = all(masks[k][:,:,i].sum() == 0 for k in ("sternum", "heart") for i in (0, -1))

    if heart_ok and sternum_ok and edges_clear:
        fov = "whole_body" if masks["sacrum"].sum() > 0 else "thorax"
    elif (
        masks["liver"].sum() > 0
        and (masks["spleen"].sum() > 0
        or masks["kidney"].sum() > 0)
        ):
        fov = "abdomen"
    else:
        fov = "unknown"
    
    # if 'tmp_' in output_path:
    #     os.system('rm -rf '+output_path)

    return fov