import numpy as np
import nibabel as nib

def check_setting(columns: list[str], no_fov: bool, no_phase: bool):
    """Check if the settings for FOV and Phase recognition are consistent with existing columns."""
    if no_fov and 'fov' in columns:
        raise ValueError("FOV recognition is disabled but 'fov' column exists in the output CSV.")
    if no_phase and 'phase' in columns:
        raise ValueError("Phase recognition is disabled but 'phase' column exists in the output CSV.")
    if not no_fov and 'fov' not in columns:
        raise ValueError("FOV recognition is enabled but 'fov' column does not exist in the output CSV.")
    if not no_phase and 'phase' not in columns:
        raise ValueError("Phase recognition is enabled but 'phase' column does not exist in the output CSV.")

def arr_union(arrs: list[np.ndarray]) -> np.ndarray:
    """Compute the union of multiple binary masks."""
    union_arr = np.zeros_like(arrs[0], dtype=bool)
    for arr in arrs:
        union_arr = np.logical_or(union_arr, arr.astype(bool))
    return union_arr.astype(np.uint8)

def read_nifti(path: str, to_ras: bool = False) -> np.ndarray:
    """Read a NIfTI file and return its data as a Nifti1Image and numpy array."""
    obj = nib.load(path)
    if to_ras:
        obj = nib.as_closest_canonical(obj)
    data = np.array(obj.get_fdata())
    return obj, data

def remove_blank_slices(image: np.ndarray):
    """Remove blank slices from a 3D image.
    A slice is considered blank if all pixels in that axial slice have the same value.
    """
    if image.ndim != 3:
        return image, slice(None)

    # per-slice min and max
    mins = image.min(axis=(0, 1))
    maxs = image.max(axis=(0, 1))

    non_blank = maxs != mins

    if not np.any(non_blank):
        return image, slice(None)

    return image[:, :, non_blank], non_blank
