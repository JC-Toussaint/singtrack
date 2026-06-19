import os
import numpy as np
import meshio

def generate_bloch_point_sol(msh_filename, sol_filename):
    # 1. Gestion des chemins absolus avec 'os'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    msh_path = os.path.join(script_dir, msh_filename)
    sol_path = os.path.join(script_dir, sol_filename)
    
    print(f"Lecture du maillage : {msh_path}")
    
    if not os.path.exists(msh_path):
        raise FileNotFoundError(f"Le fichier '{msh_filename}' est introuvable dans {script_dir}")

    # 2. Lire le maillage GMSH
    mesh = meshio.read(msh_path)
    nodes = mesh.points  # Tableau (Nnodes, 3) -> x, y, z
    
    # 3. Initialiser la matrice d'aimantation (Nnodes, 3)
    magnetization = np.zeros_like(nodes)
    
    for i, (x, y, z) in enumerate(nodes):
        # Distance 3D par rapport au centre (0,0,0)
        r_3d = np.sqrt(x**2 + y**2 + z**2)
        
        if r_3d == 0:
            # Pile-poil au centre : Singularité stricte, l'aimantation s'annule
            mx, my, mz = 0.0, 0.0, 0.0
        else:
            # Point de Bloch de type "Source" : m pointe dans la direction du vecteur position r
            # On normalise pour que la norme de l'aimantation |m| soit égale à 1 partout ailleurs
            mx = x / r_3d
            my = y / r_3d
            mz = z / r_3d
            
        magnetization[i] = [mx, my, mz]
        
    # 4. Sauvegarder dans sol.in
    np.savetxt(sol_path, magnetization, delimiter='\t', fmt='%.8e')
    print(f"Succès ! Fichier '{sol_filename}' généré avec un point de Bloch au centre.")
    print(f"Chemin du fichier : {sol_path}")

if __name__ == '__main__':
    generate_bloch_point_sol("cube.msh", "sol.in")