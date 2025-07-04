import numpy as np

def add_noise(volume, snr=1.0, seed=None):
    """
    Adds Gaussian noise to a 3D volume based on a signal-to-noise ratio.

    Parameters:
    - volume: np.ndarray, input volume
    - snr: float, desired signal-to-noise ratio (higher = less noise)
    - seed: int or None, for reproducibility

    Returns:
    - noisy_volume: np.ndarray
    """
    if seed is not None:
        np.random.seed(seed)

    signal_power = np.mean(volume**2)
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), size=volume.shape)

    noisy_volume = volume + noise
    return noisy_volume
