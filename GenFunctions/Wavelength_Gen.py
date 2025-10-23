import numpy as np

def active_wave(wavelengths, x, y, z):
        Wx = np.linspace(wavelengths[0], wavelengths[0], len(x))
        Wy = np.linspace(wavelengths[1], wavelengths[1], len(y))
        Wz = np.linspace(wavelengths[2], wavelengths[2], len(z))
        wX, wY, wZ = np.meshgrid(Wx, Wy, Wz, indexing='ij')
        return wX, wY, wZ