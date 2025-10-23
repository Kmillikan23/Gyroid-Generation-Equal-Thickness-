import numpy as np
from skimage.measure import marching_cubes
import pyvista as pv
from collections import defaultdict
from types import SimpleNamespace
from  GenFunctions.Wavelength_Gen import active_wave


def GyroidIso(invals, margin):

    length = invals.length
    wavelengths=invals.wavelengths
    rotation=invals.rotation
    target_res = invals.res
    # Create coordinate grids

    xmin, xmax = -length[0] / 2, length[0] / 2
    ymin, ymax = -length[1] / 2, length[1] / 2
    zmin, zmax = -length[2] / 2, length[2] / 2
    
    Nx = round(length[0]  / target_res)
    Ny = round(length[1] / target_res)
    Nz = round(length[2]  / target_res)

    hx = length[0]/Nx
    hy = length[1]/Ny
    hz = length[2]/Nz
    h = min(hx, hy, hz)

    Nx = int(round(length[0] / h)) + 1
    Ny = int(round(length[1] / h)) + 1
    Nz = int(round(length[2] / h)) + 1


    x = np.linspace(xmin - margin, xmax + margin,Nx)
    y = np.linspace(ymin - margin, ymax + margin, Ny)
    z = np.linspace(zmin - margin, zmax + margin, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    wX, wY, wZ = active_wave(wavelengths, x, y, z)

    theta_x,theta_y,theta_z = rotation
    degrees = True
   
    if degrees:
        ax = np.deg2rad(theta_x)
        ay = np.deg2rad(theta_y)
        az = np.deg2rad(theta_z)
    else:
        ax, ay, az = theta_x, theta_y, theta_z

    # Rotation matrices
    Rx = np.array([[1,0,0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax),  np.cos(ax)]])
    
    Ry = np.array([[ np.cos(ay), 0, np.sin(ay)],
                   [0, 1,0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az),  np.cos(az), 0],
                   [0, 0, 1]])

    # Combined rotation: first Rx, then Ry, then Rz
    R = Rz @ Ry @ Rx 

    # Stack coords and apply rotation
    coords = np.stack((X, Y, Z), axis=-1)        
    rotated = coords @ R                       
    Xr, Yr, Zr = rotated[..., 0], rotated[..., 1], rotated[..., 2]

    # Gyroid implicit function on rotated coords
    F = (
        np.cos(2 * np.pi * Xr / wX) * np.sin(2 * np.pi * Yr / wY)
      + np.cos(2 * np.pi * Yr / wY) * np.sin(2 * np.pi * Zr / wZ)
      + np.cos(2 * np.pi * Zr / wZ) * np.sin(2 * np.pi * Xr / wX)
    )
 
    verts, faces, _, _ = marching_cubes(F, level=0)

    # Map marching cubes vertex indices to the actual coordinate system, marching cibes is index space
    verts[:, 0] = np.interp(verts[:, 0], [0, len(x) - 1], [x[0], x[-1]])
    verts[:, 1] = np.interp(verts[:, 1], [0, len(y) - 1], [y[0], y[-1]])
    verts[:, 2] = np.interp(verts[:, 2], [0, len(z) - 1], [z[0], z[-1]])

    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces])
    mesh = pv.PolyData(verts, faces_pv)
    mesh = mesh.triangulate()

    if margin == 0:
        mesh = clean_mesh(mesh)
    else:
        mesh = clean_ref(mesh)

    mesh.compute_normals(point_normals=True, cell_normals=False, inplace=True)
    normals_i = np.array(mesh.point_normals)
   
    normals = normals_i / np.linalg.norm(normals_i, axis=1, keepdims=True)
    
    face = extract_faces_from_pyvista(mesh)
    vertsF = np.array(mesh.points)
    boundF = np.array(find_boundary_edges(face))
    boundV = np.unique(boundF.reshape(-1, 1))

    final = SimpleNamespace(iso_mesh=mesh, verts=vertsF, face=face, normals=normals, bounds=boundF)
    return final

def clean_mesh(mesh):
   
    mesh = mesh.triangulate()
    mesh = mesh.decimate(0.5)
    return mesh

def clean_ref(mesh):
    mesh = mesh.clean()
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    mesh = mesh.smooth(n_iter=100)
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    #mesh = mesh.decimate(0.05)
    return mesh

def extract_faces_from_pyvista(mesh):
    faces_flat = mesh.faces
    faces = []
    i = 0
    while i < len(faces_flat):
        n = faces_flat[i]  # number of vertices in this face
        face = faces_flat[i + 1: i + 1 + n]
        faces.append(face)
        i += n + 1
    return np.array(faces)

def find_boundary_edges(faces):
    edge_count = defaultdict(int)
    for face in faces:
        num_vertices = len(face)
        for i in range(num_vertices):
            edge = tuple(sorted((face[i], face[(i + 1) % num_vertices])))
            edge_count[edge] += 1
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    return boundary_edges
