import numpy as np

def apply_missing_wedge(volume, wedge_angle=60):
    """
    Applies a missing wedge mask in Fourier space to simulate limited tilt range.

    Parameters:
    - volume: 3D np.ndarray, the volume to mask
    - wedge_angle: int, full angular extent of the tilt (e.g., ±60°)

    Returns:
    - masked_volume: np.ndarray, the volume after applying the missing wedge
    """
    assert volume.ndim == 3
    fft_vol = np.fft.fftn(volume)
    fft_vol = np.fft.fftshift(fft_vol)

    Z, Y, X = volume.shape
    center_z = Z // 2
    max_tilt = np.radians(wedge_angle)

    # Build missing wedge mask (1 = keep, 0 = zero out)
    mask = np.zeros_like(fft_vol, dtype=np.float32)
    for z in range(Z):
        angle = np.abs(np.arctan2(z - center_z, Y))  # angle from center
        mask[z, :, :] = angle <= max_tilt / 2

    fft_masked = fft_vol * mask
    fft_masked = np.fft.ifftshift(fft_masked)
    volume_masked = np.real(np.fft.ifftn(fft_masked))

    return volume_masked
