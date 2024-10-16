import numpy as np

def add_noise(image, desired_snr_db, seed):
    signal_power = np.mean(image ** 2)
    noise_power = signal_power / desired_snr_db

    mean = 0
    std = np.sqrt(noise_power)
    np.random.seed(seed)
    gaussian_noise = np.random.normal(mean, std, image.shape)

    noisy_image = image + gaussian_noise
    noisy_image_uint8 = np.uint8(np.clip(noisy_image, 0, 255))

    return noisy_image_uint8

def Fourier_Trans(img):
    f_transform = np.fft.fft2(img)
    f_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shifted) + 1)
    return f_shifted, magnitude_spectrum

def low_pass_filter(img, radius):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    low_pass = np.zeros((rows, cols), dtype=np.uint8)
    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((X - ccol) ** 2 + (Y - crow) ** 2)
    mask = dist_from_center <= radius
    low_pass[mask] = 1
    return low_pass