# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def apply_one_mask(x, mask):
    mask_keep = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
    return torch.gather(x, dim=1, index=mask_keep)

def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep

    """
    all_x = []
    for m in masks:
        all_x .append(apply_one_mask(x, m))
    return torch.cat(all_x, dim=0)


def visualize_mask(mask_indices, grid_size=(14, 14)):
    """
    Convert mask indices into a binary grid and display it
    Args:
        mask_indices: Tensor of indices
        grid_size: Tuple of (height, width) for the grid
    """
    # Create empty grid
    total_patches = grid_size[0] * grid_size[1]
    mask = torch.zeros(total_patches)
    
    # Fill in the masked positions with 1s
    mask[mask_indices] = 1
    
    # Reshape to grid
    mask_grid = mask.reshape(grid_size)
    
    return mask_grid

def plot_masks(enc_mask, pred_mask, grid_size=(14, 14)):
    """
    Plot encoder and predictor masks side by side
    """
    import matplotlib.pyplot as plt
    
    enc_grid = visualize_mask(enc_mask, grid_size)
    pred_grid = visualize_mask(pred_mask, grid_size)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(enc_grid, cmap='binary')
    ax1.set_title('Encoder Mask')
    ax1.axis('off')
    
    ax2.imshow(pred_grid, cmap='binary')
    ax2.set_title('Predictor Mask')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()