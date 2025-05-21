import torch
import matplotlib.pyplot as plt
import numpy as np

def create_masked_spectrogram(spectrogram, patch_t, patch_mel, patch_indices=None, show_colorbar=False):
    """
    Visualizes a spectrogram with a transparent black overlay on NON-masked patches,
    overlaid on a darker color spectrogram (plasma cmap), and dashed patch borders.

    Args:
        spectrogram (torch.Tensor): Spectrogram tensor of shape (n_time_bins, n_mels) or (C, n_time_bins, n_mels).
        patch_t (int): Patch dimension in the time axis.
        patch_mel (int): Patch dimension in the mel axis.
        patch_indices (torch.Tensor, optional): Tensor of patch indices to be MASKED (black overlay). If None, plot the entire spectrogram in color. Defaults to None.
        show_colorbar (bool, optional): Whether to display the colorbar. Defaults to True.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure of the spectrogram visualization.
    """
    if spectrogram.ndim == 3:
        spectrogram = spectrogram.mean(dim=0) # Average channels if multi-channel

    n_time_bins, n_mels = spectrogram.shape

    n_patches_t = n_time_bins // patch_t
    n_patches_mel = n_mels // patch_mel
    total_patches = n_patches_t * n_patches_mel

    fig, ax = plt.subplots()

    # Plot the base spectrogram in color ALWAYS, using a darker cmap (plasma)
    im_base = ax.imshow(spectrogram.cpu().numpy().T, origin='lower', aspect='equal', cmap='inferno') # Transpose and use plasma cmap

    if patch_indices is not None:
        # Create a transparent BLACK overlay for NON-masked patches (inverted mask)
        overlay_mask = np.zeros((n_mels, n_time_bins, 4))
        # Initialize overlay to semi-transparent BLACK for ALL patches initially
        overlay_mask[:, :, :] = [0, 0, 0, 0.5] # Black with 50% alpha

        valid_patch_indices = []
        for patch_idx in patch_indices:
            if 0 <= patch_idx < total_patches:
                valid_patch_indices.append(patch_idx)
            else:
                print(f"Warning: Patch index {patch_idx} is out of range [0, {total_patches}). Ignoring it.")

        # Make MASKED patches fully VISIBLE by setting alpha to 0 in the overlay for these patches
        for patch_idx in valid_patch_indices:
            # Corrected patch index calculation for row-major (mel-major in spectrogram context)
            patch_row_idx = patch_idx % n_patches_mel  # mel patch index (y-axis) - faster varying
            patch_col_idx = patch_idx // n_patches_mel # time patch index (x-axis) - slower varying

            start_mel = patch_row_idx * patch_mel
            end_mel = start_mel + patch_mel
            start_time = patch_col_idx * patch_t
            end_time = start_time + patch_t

            # Set MASKED patch region in overlay to be FULLY TRANSPARENT (alpha=0), making base spectrogram visible
            overlay_mask[start_mel:end_mel, start_time:end_time, 3] = 0 # Set alpha channel to 0 for masked patches

        ax.imshow(overlay_mask, origin='lower', aspect='equal') # Overlay the transparent patches


    if show_colorbar: # Conditional colorbar display
        fig.colorbar(im_base, ax=ax, format='%+2.0f dB') # Add colorbar for the base color spectrogram


    # Add dashed white patch border lines
    time_locations = np.arange(patch_t, n_time_bins, patch_t)
    mel_locations = np.arange(patch_mel, n_mels, patch_mel)

    ax.vlines(time_locations - 0.5, ymin=-0.5, ymax=n_mels - 0.5, color='white', linewidth=1.5, linestyle='--') # Vertical lines, Dashed White
    ax.hlines(mel_locations - 0.5, xmin=-0.5, xmax=n_time_bins - 0.5, color='white', linewidth=1.5, linestyle='--') # Horizontal lines, Dashed White


    # ax.set_title('Spectrogram with Masked Patches and Patch Borders')
    ax.set_xlabel('Time Bins')
    ax.set_ylabel('Mel Bins')
    return fig

if __name__ == '__main__':
    # Example Usage
    import torchaudio.transforms as T

    # Generate a dummy spectrogram (you can replace this with your actual spectrogram)
    n_time_bins = 256 # Increased time bins for better visualization
    n_mels = 128 # Increased mel bins for better visualization
    spectrogram_data = torch.randn(n_time_bins, n_mels)
    spectrogram_dB = 20 * torch.log10(torch.relu(spectrogram_data) + 1e-6) # Example dB conversion

    # Spectrogram with channels
    spectrogram_channels = torch.randn(2, n_time_bins, n_mels)
    spectrogram_channels_dB = 20 * torch.log10(torch.relu(spectrogram_channels) + 1e-6) # Example dB conversion

    patch_t = 32
    patch_mel = 8

    n_patches_t = n_time_bins // patch_t
    n_patches_mel = n_mels // patch_mel
    total_patches = n_patches_t * n_patches_mel
    print(f'{total_patches=}')

    # Example 1: Plot the entire spectrogram in color, no colorbar
    fig1 = create_masked_spectrogram(spectrogram_dB, patch_t, patch_mel, patch_indices=None, show_colorbar=False)
    fig1.suptitle("Entire Spectrogram with Borders (Plasma Color, No Colorbar)")
    plt.show()

    # Example 2: Plot spectrogram with some masked patches (black overlay on NON-masked), now with COLOR colorbar and transparent patches
    patch_indices_to_mask = torch.tensor([0, 1, 2, 3, n_patches_mel, n_patches_mel + 1, 2*n_patches_mel, 2*n_patches_mel + 2, total_patches -1, total_patches - n_patches_mel]) # Example patch indices, including out-of-range index
    fig2 = create_masked_spectrogram(spectrogram_dB, patch_t, patch_mel, patch_indices=patch_indices_to_mask, show_colorbar=True) # show_colorbar=True is default now for masked case
    fig2.suptitle("Spectrogram with Transparent BLACK Overlay on NON-Masked Patches and Borders (Plasma Spectrogram, Transparent Black Overlay, COLOR Colorbar)")
    plt.show()

    # Example 3: Spectrogram with channels, with colorbar (default)
    fig3 = create_masked_spectrogram(spectrogram_channels_dB, patch_t, patch_mel, patch_indices=patch_indices_to_mask) # show_colorbar=True is default
    fig3.suptitle("Spectrogram with Channels, Transparent BLACK Overlay on NON-Masked Patches and Borders (Channels Averaged, With COLOR Colorbar)")
    plt.show()