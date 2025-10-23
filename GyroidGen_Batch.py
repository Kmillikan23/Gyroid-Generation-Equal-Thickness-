
import os
from GenFunctions.Gyroid_Thicken import NotCorrectedThick
from GenFunctions.Gyroid_Isosurface import GyroidIso

from types import SimpleNamespace
import pyvista as pv
from scipy.optimize import root_scalar
import pandas as pd


# fixed parameters
length         = (10,10,10)
res            = 0.2
#target_porosity = 0.85


# read input spreadsheet
file_location = r"C:\Users\kmillikan\OneDrive - Oceanit-GCC High\Project Work\Gyroid Generation"  
input_filename = "CurrentGenAll.xlsx"          

output_dir = "output_STLs"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel(os.path.join(file_location, input_filename))

results = []

for _, row in df.iterrows():
    ID         = row['ID']
    wavelengths = (row['wavex'], row['wavey'], row['wavez'])
    rotation    = (row['thetax'], row['thetay'], row['thetaz'])
    target_porosity = (row['porosity'])
    
    params = SimpleNamespace(
        length     = length,
        wavelengths= wavelengths,
        rotation   = rotation,
        res        = res,
        thickness  = None
    )

    def compute_porosity(thk):
        params.thickness = thk
        iso   = GyroidIso(params, 0)
        mesh  = NotCorrectedThick(params, iso).connectivity(extraction_mode="largest")
        vol   = mesh.volume
        por   = 1.0 - vol / (length[0]*length[1]*length[2])
        return por, mesh

    # root-finding function
    f = lambda t: compute_porosity(t)[0] - target_porosity

  
    a, b = 0.2, 0.8
    fa, fb = f(a), f(b)
    while fa * fb > 0:
        b *= 1.25
        fb = f(b)

    sol = root_scalar(f, bracket=[a, b], method='brentq', xtol=1e-2)
    thickness_opt = sol.root

    # compute final mesh & save
    _, mesh_opt = compute_porosity(thickness_opt)
    filename = f"Gyroid_{ID}.stl"
    stlsave = os.path.join(output_dir, filename)
    mesh_opt.save(stlsave)


    
    results.append({'ID': ID, 'thickness': thickness_opt})

# merge thickness results back into the original table
out_df = df.merge(pd.DataFrame(results), on='ID')
out_df.to_excel("output_with_thickness.xlsx", index=False)

print("Done: generated STLs and wrote output_with_thickness.xlsx")
