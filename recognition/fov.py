import os
import numpy as np
from typing import Literal
from os.path import join
from nibabel.nifti1 import Nifti1Image
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.map_to_binary import class_map

from recognition.utils import arr_union, read_nifti, remove_blank_slices

def read_mask(path: str, organ: str, non_blank = slice(None)) -> np.ndarray:
    """Read the segmentation mask for a specific organ."""
    if organ == "kidney":
        mask = arr_union([read_nifti(join(path,f"{part}.nii.gz"))[1]
                        for part in ["kidney_left", "kidney_right"]])
    else:
        _, mask = read_nifti(join(path,f"{organ}.nii.gz"))
    return mask[:,:,non_blank]

def fov_recon(image_obj: Nifti1Image, image: np.ndarray, im_type: Literal["MRI", "CT"], output_path: str) -> str:
    """
    Determine the anatomical field of view (FOV) of `image_path`.

    Arguments:
    - image_obj (Nifti1Image): Nifti1Image object of the image.
    - image (np.ndarray): Numpy array of the image data.
    - im_type (str): Type of the image ('MRI' or 'CT').
    - output_path (str): Path to save segmentation masks.

    Returns:
    - str: Recognized FOV value.
    """
    image, non_blank_slices = remove_blank_slices(image)
    if image.shape[2] < 10:
        return "scout"
    
    task = "total" if im_type == "CT" else "total_mr"
    organs = ["heart", "sternum", "sacrum", "liver", "spleen", "kidney"]
    if im_type == "MRI":
        organs.remove("sternum")  # sternum not available in total_mr model
        organs.append("prostate")
    
    no_saving = output_path is None
    segmentation = totalsegmentator(image_obj, output_path, ml=True, skip_saving=no_saving, task=task)
    mask_data = np.array(segmentation.get_fdata())
    
    task_map = class_map.get(task, {})
    task_map_rev = {v: k for k, v in task_map.items()}
    masks = {}
    for o in organs:
        mask = mask_data.copy()
        if o == "kidney":
            mask[(mask_data != task_map_rev["kidney_left"]) *
                 (mask_data != task_map_rev["kidney_right"]) ] = 0
        else:
            mask[mask_data != task_map_rev[o]] = 0
        masks[o] = mask[:,:,non_blank_slices]

    heart_ok = masks["heart"].sum() > 0
    if im_type == "MRI":
        sternum_ok = True
        edges_clear = all(masks["heart"][:,:,i].sum() == 0 for i in (0, 1, -1, -2)) # Two clear slices because sometimes last one is not segmented
    else:
        sternum_ok = masks["sternum"].sum() > 0
        edges_clear = all(masks[k][:,:,i].sum() == 0 for k in ("sternum", "heart") for i in (0, 1, -1, -2))

    if heart_ok and sternum_ok and edges_clear:
        if masks["sacrum"].sum() > 0:
            if im_type == "MRI" and masks["liver"].sum() == 0:
                fov = "spine"
            else:
                fov = "whole_body"
        elif masks["liver"].sum() > 0 and masks["liver"][:,:,0].sum() == 0:
             fov = "thorax_abdomen"
        else:
            fov = "thorax"
    elif (
        masks["liver"].sum() > 0
        and (masks["spleen"].sum() > 0
        or masks["kidney"].sum() > 0)
        ):
        fov = "abdomen"
    elif (
        im_type == "MRI"
        and masks["prostate"].sum() > 0
        and all(masks["prostate"][:,:,i].sum() == 0 for i in (0, 1, -1, -2))
        ):
        fov = "pelvis"
    else:
        fov = "unknown"
    
    return fov