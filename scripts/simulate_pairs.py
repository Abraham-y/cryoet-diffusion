import numpy as np
import os
from Bio.PDB import PDBParser, MMCIFParser
from scipy.ndimage import gaussian_filter, rotate
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.wedge import apply_missing_wedge
from utils.noise import add_noise
from scipy.fftpack import fft, ifft, fftfreq

def load_structure(path_to_structure, voxel_size=2.0, volume_size=160, sigma=1.0):
    """
    Load a PDB or mmCIF file, place atoms into a 3D grid using Gaussian densities,
    and return a normalized volume.
    """
    extension = os.path.splitext(path_to_structure)[-1].lower()
    if extension == ".pdb":
        parser = PDBParser(QUIET=True)
    elif extension == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError("Unsupported file format: must be .pdb or .cif")

    structure = parser.get_structure("molecule", path_to_structure)

    # Extract heavy atoms only
    atom_coords = [
        atom.coord for atom in structure.get_atoms()
        if atom.element not in {"H", "D"}
    ]

    atom_coords = np.array(atom_coords)
    if len(atom_coords) == 0:
        raise ValueError("No heavy atoms found.")

    # Center and convert to voxel space
    center = atom_coords.mean(axis=0)
    coords_centered = atom_coords - center
    max_range = voxel_size * volume_size / 2
    coords_voxel = ((coords_centered + max_range) / voxel_size).astype(int)

    # Keep atoms within volume
    valid_mask = np.all((coords_voxel >= 0) & (coords_voxel < volume_size), axis=1)
    coords_voxel = coords_voxel[valid_mask]

    # Fill volume
    volume = np.zeros((volume_size, volume_size, volume_size), dtype=np.float32)
    for x, y, z in coords_voxel:
        volume[x, y, z] += 1.0

    volume = gaussian_filter(volume, sigma=sigma)
    volume = volume / volume.max() if volume.max() > 0 else volume

    return volume


def simulate_tilt_series(volume, tilt_range=(-60, 60), step=2):
    """
    Simulates a tilt series by projecting a 3D volume at different angles.

    Parameters:
    - volume: 3D numpy array (Z, Y, X)
    - tilt_range: tuple, min and max tilt angle in degrees
    - step: int, angle step size in degrees

    Returns:
    - tilt_series: np.ndarray of shape (num_tilts, Y, X)
    - angles: list of tilt angles used
    """
    angles = np.arange(tilt_range[0], tilt_range[1] + step, step)
    projections = []

    for angle in angles:
        # Rotate around Y axis (simulate stage tilt)
        rotated = rotate(volume, angle, axes=(0, 2), reshape=False, order=1, mode='constant', cval=0.0)
        projection = np.sum(rotated, axis=0)  # Project along Z-axis (transmission direction)
        projections.append(projection)

    tilt_series = np.stack(projections, axis=0)
    return tilt_series, angles

def filtered_backprojection(tilt_series, angles, volume_shape):
    """
    Reconstructs a 3D volume from a tilt series using filtered backprojection (Ram-Lak filter).

    Parameters:
    - tilt_series: np.ndarray of shape (num_angles, height, width)
    - angles: list or np.ndarray of tilt angles in degrees
    - volume_shape: tuple (Z, Y, X) specifying output volume shape

    Returns:
    - recon_volume: np.ndarray of shape (Z, Y, X), the reconstructed volume
    """
    num_angles, height, width = tilt_series.shape
    recon_volume = np.zeros(volume_shape, dtype=np.float32)

    # Prepare Ram-Lak filter in frequency domain
    freqs = fftfreq(width).reshape(1, -1)  # shape: (1, width)
    ram_lak = np.abs(freqs)  # shape: (1, width)

    for i, angle in enumerate(angles):
        projection = tilt_series[i]  # shape: (height, width)

        # Apply Ram-Lak filter to each row
        proj_fft = fft(projection, axis=1)
        proj_filtered = np.real(ifft(proj_fft * ram_lak, axis=1))

        # Expand filtered projection into 3D by tiling along z-axis
        backproj = np.tile(proj_filtered[np.newaxis, :, :], (volume_shape[0], 1, 1))

        # Rotate 3D backprojection volume around y-axis
        rotated = rotate(backproj, angle, axes=(0, 2), reshape=False, order=1, mode='constant', cval=0.0)

        # Accumulate backprojected volume
        recon_volume += rotated

    # Normalize result
    recon_volume /= len(angles)
    recon_volume = recon_volume / np.max(recon_volume) if np.max(recon_volume) > 0 else recon_volume

    return recon_volume

def create_training_pair(pdb_path, out_dir, snr=0.1, use_projection=True):
    """
    Loads a structure, simulates a tilt series, reconstructs from it (with noise and missing wedge),
    and saves both clean and corrupted volumes.
    
    Args:
        pdb_path (str): Path to the PDB or CIF file.
        out_dir (str): Directory to save outputs in "clean/" and "corrupted/" folders.
        snr (float): Signal-to-noise ratio for simulated noise.
        use_projection (bool): Whether to simulate and reconstruct from tilt series.
    """
    os.makedirs(os.path.join(out_dir, "clean"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "corrupted"), exist_ok=True)

    vol = load_structure(pdb_path, voxel_size=2, volume_size=160)
    clean = vol / np.max(vol)

    name = os.path.splitext(os.path.basename(pdb_path))[0]

    if use_projection:
        # Step 1: simulate tilt series
        tilt_series, angles = simulate_tilt_series(clean, tilt_range=(-60, 60), step=2)

        # Step 2: add noise to the projections
        noisy_tilts = np.array([add_noise(p, snr=snr) for p in tilt_series])

        # Step 3: reconstruct from noisy tilt series
        corrupted = filtered_backprojection(noisy_tilts, angles, volume_shape=clean.shape)
    else:
        # Fallback: direct corruption in volume space (not realistic)
        corrupted = apply_missing_wedge(clean)
        corrupted = add_noise(corrupted, snr=snr)

    # Save both volumes
    np.save(os.path.join(out_dir, "clean", f"{name}_clean.npy"), clean)
    np.save(os.path.join(out_dir, "corrupted", f"{name}_corrupted.npy"), corrupted)

    print(f"✅ Saved training pair for {name} in {out_dir}")


def test_cryoet_pipeline():
    """
    Tests the entire cryo-ET preprocessing pipeline with a known small structure.
    Verifies that the clean and corrupted volumes are correctly generated.
    """
    import tempfile
    import shutil
    import numpy as np

    print("🧪 Running test_cryoet_pipeline...")

    # Use a known test file (change this if needed)
    test_path = "data/pdbs/1oel.pdb"  # Make sure this file exists
    temp_dir = tempfile.mkdtemp()

    try:
        # Run the pipeline
        create_training_pair(test_path, temp_dir, snr=0.05, use_projection=True)

        base_name = os.path.splitext(os.path.basename(test_path))[0]
        clean_path = os.path.join(temp_dir, "clean", f"{base_name}_clean.npy")
        corrupted_path = os.path.join(temp_dir, "corrupted", f"{base_name}_corrupted.npy")

        clean = np.load(clean_path)
        corrupted = np.load(corrupted_path)

        assert clean.shape == corrupted.shape == (160, 160, 160), "Volume shape mismatch"
        assert 0 <= clean.max() <= 1, "Clean volume not normalized"
        assert 0 <= corrupted.max() <= 1, "Corrupted volume not normalized"
        assert not np.allclose(clean, corrupted), "Corrupted volume too similar to clean one"

        print("✅ test_cryoet_pipeline passed.")

    except Exception as e:
        print("❌ test_cryoet_pipeline failed:", e)

    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    # test_cryoet_pipeline()
    create_training_pair("data/pdbs/1oel.pdb", "data/simulated")
    create_training_pair("data/pdbs/4v6x.cif", "data/simulated")

