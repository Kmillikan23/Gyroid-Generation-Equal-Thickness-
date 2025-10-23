from GenFunctions.ThicknessVal_Gen import thicknessVal
from  GenFunctions.Gyroid_Isosurface import GyroidIso
import numpy as np
import pyvista as pv

def NotCorrectedThick(inval,iso):
    middle_verts = iso.verts
    middle_face = iso.face
    normals = iso.normals
    boundF = iso.bounds

    t_array_mesh = thicknessVal(iso.normals,inval.thickness)

    v_top = middle_verts + normals * t_array_mesh
    v_bot = middle_verts - normals * t_array_mesh
    f_bot = middle_face
    f_top = middle_face

    wallF = np.zeros((2 * len(boundF), 3))
    wallV = np.zeros((4 * len(boundF), 3))

    for i in range(1, len(boundF) + 1):
        i_v1 = (4 * i - 4)
        i_v2 = (4 * i - 3)
        i_v3 = (4 * i - 2)
        i_v4 = (4 * i - 1)
        wallV[i_v1, :] = v_bot[boundF[i - 1, 0], :]
        wallV[i_v2, :] = v_top[boundF[i - 1, 0], :]
        wallV[i_v3, :] = v_bot[boundF[i - 1, 1], :]
        wallV[i_v4, :] = v_top[boundF[i - 1, 1], :]
        wallF[2 * i - 2, :] = [i_v1, i_v2, i_v4]
        wallF[2 * i - 1, :] = [i_v3, i_v1, i_v4]

    faces_bot_pv = np.hstack([np.full((f_bot.shape[0], 1), 3, dtype=np.int64), f_bot])
    faces_top_pv = np.hstack([np.full((f_top.shape[0], 1), 3, dtype=np.int64), f_top])
    faces_wall_pv = np.hstack([np.full((wallF.shape[0], 1), 3, dtype=np.int64), wallF])
    faces_wall_pv = faces_wall_pv.astype(int)

    mesh_bot = pv.PolyData(v_bot, faces_bot_pv)
    mesh_top = pv.PolyData(v_top, faces_top_pv)
    mesh_wall = pv.PolyData(wallV, faces_wall_pv)

    final = pv.merge([mesh_bot, mesh_wall, mesh_top])
    final = final.clean()
    if not final.is_all_triangles:
        final = final.triangulate()
    final = final.clean()                 # remove duplicates, unused points
    final = final.triangulate()           # ensure pure triangles
    final.compute_normals(               # make all normals point outwards
    auto_orient_normals=True,
    flip_normals=False,
    inplace=True
)
    return final

def Thicken(inval):
    ref_iso = GyroidIso(inval, 1)
    tref = thicknessVal(ref_iso.normals,inval.thickness)