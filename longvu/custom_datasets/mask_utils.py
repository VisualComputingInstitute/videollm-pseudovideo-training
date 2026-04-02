import cv2
import random
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Iterable
from einops import pack, repeat, rearrange
from os import PathLike

def read_BGR_image(img_path: PathLike) -> NDArray:
    """Load the image located at img_path and return in as an H x W x 3
    numpy array, with channels in BGR order.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if(img is None):
        raise Exception("Opencv tried to access invalid image {0}".format(
            img_path))    
    return img 

def convert_png_mask_to_id_map(bgr_png_mask: NDArray) -> NDArray:
    """Converts a COCO-formatted H x W x 3 png mask (channel order BGR,
    dtype uint8) to an HxW integer array whose (i, j)-th element 
    contains the segment id for the corresponding pixel of the mask.
    """
    bgr_png_mask = np.ndarray.astype(bgr_png_mask, np.int32)
    return np.ndarray.astype(bgr_png_mask[:, :, 0]*256*256 
                             + bgr_png_mask[:, :, 1]*256
                             + bgr_png_mask[:, :, 2], np.int32)

def convert_id_map_to_binary_masks(id_map: NDArray) -> Tuple[NDArray, NDArray]:
    """Given an H x W integer array whose (i, j)-th element contains the
    segment id for the corresponding pixel, returns a tuple whose first
    element is the sorted segment ids and second element is an H x W x N 
    boolean array such that:
    slice[:, :, 0] contains True for all void-labeled pixels, False
    otherwise
    For i>0, slice [:, :, i] contains True for all
    pixels that are part of the i-th segment id when ids are sorted in
    ascending order, and False otherwise.
    """
    segment_ids = np.sort(np.unique(id_map))
    binary_masks = [id_map == _ for _ in segment_ids]
    if(len(binary_masks) == 0):
        return binary_masks
    
    binary_masks, _ = pack(binary_masks, "h w *")
    assert(binary_masks.shape[2] == len(segment_ids))
    return segment_ids, binary_masks

def create_random_relative_path(max_path_steps: int) -> NDArray:
    num_path_steps = random.randint(1, max_path_steps)
    path_dxys = np.random.uniform(-0.4, 0.4, (num_path_steps, 2))
    path_points = np.random.uniform(0.3, 0.7, (1, 2))
    path_points = np.concatenate((path_points, path_points + np.cumsum(
        path_dxys, axis=0)), axis=0)
    #path_points = np.zeros_like(path_points)

    return path_points

def relative_time_to_frame(rel_time: float, num_frames: int) -> int:
    """Converts a relative time in [0, 1] to an absolute frame index in
    the range [0, num_frames - 1]."""
    return int(rel_time*(num_frames - 1))

def relative_bbox_to_absolute(
        top_left: Tuple[int, int], 
        bbox_h_w: Tuple[int, int], 
        img_h_w: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    top_left = (int(img_h_w[1]*top_left[0]), int(img_h_w[0]*top_left[1]))
    bbox_h_w = (int(img_h_w[1]*bbox_h_w[0]), int(img_h_w[0]*bbox_h_w[1]))
    return top_left, bbox_h_w

def relative_path_to_absolute(
        rel_path_dxys: NDArray, img_h_w: Tuple[int, int]) -> NDArray:
    return rel_path_dxys*img_h_w
    
def interpolate_path(path_xys: NDArray, rel_start_time: float, 
                     rel_end_time: float, num_frames: int) -> NDArray:
    existing_times = np.linspace(
        rel_start_time, rel_end_time, path_xys.shape[0])    
    times_to_interp = np.linspace(
        rel_start_time, rel_end_time, num_frames)
    interpolated_path = np.stack((
        np.interp(times_to_interp, existing_times, path_xys[:, 0]),
        np.interp(times_to_interp, existing_times, path_xys[:, 1])), axis=1)
    return interpolated_path

def pad_imgs(imgs: NDArray, pad_r_t_l_b: Tuple[int, int, int, int]) -> NDArray:
    """Pads each image of `imgs` (shape: (num_imgs, H, W, C)) with a 
    padding of `pad_r_t_l_b` on the right, top, left and bottom sides in 
    this order."""
    NUM_IMGS, H, W, C = imgs.shape
    assert len(pad_r_t_l_b) == 4
    if(all(_ == 0 for _ in pad_r_t_l_b )):
        return imgs.copy()
    pad_right = pad_r_t_l_b[0]
    pad_top = pad_r_t_l_b[1]
    pad_left = pad_r_t_l_b[2]
    pad_bottom = pad_r_t_l_b[3]
    padded_imgs = np.zeros(
        (imgs.shape[0], imgs.shape[1] + pad_top + pad_bottom,
        imgs.shape[2] + pad_left + pad_right, imgs.shape[3]), dtype=imgs.dtype)
    first_bottom_pad_row =padded_imgs.shape[1] - pad_bottom
    first_right_pad_col = padded_imgs.shape[2] - pad_right
    padded_imgs[:, pad_r_t_l_b[1]:first_bottom_pad_row, 
                pad_r_t_l_b[2]:first_right_pad_col] = imgs
    return padded_imgs

def compute_pad_and_inds_for_moving_mask(
        img_h_w: Tuple[int, int],
        mask_h_w: Tuple[int, int], 
        path_xys: NDArray) -> Tuple[Tuple[int, int, int, int], NDArray]:
    
    horizontal_pos = path_xys[:, 0]
    min_x = np.min(horizontal_pos).item()
    max_x = np.max(horizontal_pos).item()
    vertical_pos = path_xys[:, 1]
    min_y = np.min(vertical_pos).item()
    max_y = np.max(vertical_pos).item()
    #pad if any part of any mask would be outside the image
    pad_left = max(-min_x, 0)
    pad_right = max(max_x + mask_h_w[1] - img_h_w[1], 0)
    pad_top = max(-min_y, 0)
    pad_bottom = max(max_y + mask_h_w[0] - img_h_w[0], 0)    
    #translate to new coordinates
    new_horizontal_pos = horizontal_pos + pad_left
    new_vertical_pos = vertical_pos + pad_top
    assert np.all(new_horizontal_pos >= 0)
    assert np.all(new_vertical_pos >= 0)
    #calculate the indices that mask (effectively bbox) pixels will occupy
    all_inds = [np.meshgrid(
        np.arange(h, h + mask_h_w[1]),
        np.arange(v, v + mask_h_w[0])) 
        for v, h in zip(new_vertical_pos, new_horizontal_pos)]
    all_inds = np.stack(
        [np.stack((np.ravel(_[1]), np.ravel(_[0])), axis=1) for _ in all_inds],
        axis=0)
    assert all_inds.shape == (path_xys.shape[0], mask_h_w[0]*mask_h_w[1], 2)

    assert (pad_right >= 0 and pad_top >= 0 
            and pad_left >= 0 and pad_bottom >= 0)
    return (pad_right, pad_top, pad_left, pad_bottom), all_inds

def draw_mask_on_imgs(
        imgs: NDArray, 
        row_col_indices_per_img: Tuple[NDArray, NDArray],
        mask_contents: NDArray,
        original_mask: NDArray) -> NDArray:
    flattened_mask = rearrange(
        repeat(original_mask, "H W -> FR H W RGB", FR=imgs.shape[0],
               RGB=3), 
        "FR H W RGB -> FR (H W) RGB")
    flattened_contents = rearrange(
        repeat(mask_contents, "H W RGB -> FR H W RGB", FR=imgs.shape[0]), 
        "FR H W RGB -> FR (H W) RGB")
    num_elems = flattened_mask.shape[1]
    inds = (repeat(np.arange(imgs.shape[0]), "FR -> FR N", N=num_elems), 
         row_col_indices_per_img[:, :, 0], 
         row_col_indices_per_img[:, :, 1], slice(None))
    imgs[inds] = np.where(flattened_mask, flattened_contents, imgs[inds])
     #        flattened_contents[flattened_mask])
    
    return imgs

def unpad_imgs(padded_imgs: NDArray, 
               pad_r_t_l_b: Tuple[int, int, int, int]) -> NDArray:
    if(all(_ == 0 for _ in pad_r_t_l_b)):
        return padded_imgs.copy()
    pad_right = pad_r_t_l_b[0]
    pad_top = pad_r_t_l_b[1]
    pad_left = pad_r_t_l_b[2]
    pad_bottom = pad_r_t_l_b[3]
    first_bottom_pad_row =padded_imgs.shape[1] - pad_bottom
    first_right_pad_col = padded_imgs.shape[2] - pad_right 
    return padded_imgs[:, pad_top:first_bottom_pad_row, 
                       pad_left:first_right_pad_col, :]

def draw_mask_on_path(
        imgs: NDArray,
        mask_h_w: Tuple[int, int],
        path_xys: NDArray,
        mask_bbox_contents: NDArray,
        original_bbox_mask: NDArray) -> NDArray:
    if(path_xys.shape[0] != imgs.shape[0]):
        raise ValueError("Num. imgs should equal num. path steps")
    pad_r_t_l_b, padded_inds = compute_pad_and_inds_for_moving_mask(
        imgs.shape[1:3], mask_h_w, path_xys)
    padded_imgs = pad_imgs(imgs, pad_r_t_l_b)
    padded_imgs= draw_mask_on_imgs(
        padded_imgs, padded_inds, mask_bbox_contents, original_bbox_mask)

    return unpad_imgs(padded_imgs, pad_r_t_l_b)