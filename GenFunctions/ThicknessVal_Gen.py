import numpy as np

def thicknessVal(normals, thickness):
    return np.zeros((len(normals), 3)) + thickness / 2