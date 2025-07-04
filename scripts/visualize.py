import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import napari

def show_volume_slices(volume, title="Volume Slices"):
    """Display central XY, XZ, YZ slices using matplotlib."""
    mid = volume.shape[0] // 2
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(volume[mid, :, :], cmap='gray')
    axs[0].set_title("XY Slice")
    axs[1].imshow(volume[:, mid, :], cmap='gray')
    axs[1].set_title("XZ Slice")
    axs[2].imshow(volume[:, :, mid], cmap='gray')
    axs[2].set_title("YZ Slice")

    for ax in axs:
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def save_volume_gif(volume, path="volume.gif", axis=0):
    """Export volume slices along a given axis to a .gif file."""
    slices = [volume.take(i, axis=axis) for i in range(volume.shape[axis])]
    images = [(255 * (slc / slc.max())).astype(np.uint8) for slc in slices]
    imageio.mimsave(path, images, fps=10)
    print(f"✅ Saved GIF to {path}")

def napari_view_volume(volume, title="Volume Viewer"):
    """Launch an interactive Napari viewer for a 3D volume."""
    viewer = napari.view_image(volume, name=title, colormap="gray", scale=[1, 1, 1])
    napari.run()

def visualize_volume(filepath):
    """Load a .npy 3D volume and visualize it via 2D slices, GIF, and Napari."""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    if filepath.endswith(".npy"):
        volume = np.load(filepath)
    else:
        print("❌ Unsupported format. Use .npy volumes.")
        return

    print(f"✅ Loaded volume: {filepath} | shape: {volume.shape}")

    # Show central slices
    show_volume_slices(volume, title=os.path.basename(filepath))

    # Save as GIF
    gif_path = filepath.replace(".npy", ".gif")
    save_volume_gif(volume, path=gif_path)

    # Launch Napari
    napari_view_volume(volume, title=os.path.basename(filepath))


if __name__ == "__main__":
    # Example usage
    example_path = "data/simulated/clean/1oel_clean.npy"
    visualize_volume(example_path)
    example_path = "data/simulated/clean/4v6x_clean.npy"
    visualize_volume(example_path)
    example_path = "data/simulated/corrupted/1oel_corrupted.npy"
    visualize_volume(example_path)
