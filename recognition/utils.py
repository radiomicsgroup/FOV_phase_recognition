import numpy as np
import nibabel as nib


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
