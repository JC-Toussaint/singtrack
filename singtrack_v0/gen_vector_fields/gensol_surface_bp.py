import os
import numpy as np
import meshio

def generate_surface_bloch_point_sol(msh_filename, sol_filename):
    # 1. Gestion des chemins absolus avec le module 'os'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    msh_path = os.path.join(script_dir, msh_filename)
    sol_path = os.path.join(script_dir, sol_filename)
    
    print(f"Recherche du maillage à l'adresse : {msh_path}")
    
    if not os.path.exists(msh_path):
        raise FileNotFoundError(f"Le fichier '{msh_filename}' est introuvable.")

    # 2. Lire le maillage
    mesh = meshio.read(msh_path)
    nodes = mesh.points  # Tableau (Nnodes, 3) -> x, y, z
    
    # Trouver la coordonnée Z de la surface supérieure du cube
    z_max = np.max(nodes[:, 2])
    
    # 3. Initialiser la matrice d'aimantation (Nnodes, 3)
    magnetization = np.zeros_like(nodes)
    R_core = 10.0  # Rayon de la zone de transition en nm
    
    # Position de la singularité de surface (Centre de la face supérieure du cube)
    xs, ys, zs = 0.0, 0.0, z_max
    
    for i, (x, y, z) in enumerate(nodes):
        # Vecteur reliant le nœud actuel au point de Bloch de surface
        dx = x - xs
        dy = y - ys
        dz = z - zs
        
        dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if dist_3d == 0:
            # Pile sur la singularité, l'aimantation est nulle
            mx, my, mz = 0.0, 0.0, 0.0
        else:
            # Profil de type Hérisson (source 3D) localisé 
            # L'aimantation s'aligne avec le vecteur radial sortant du point (xs, ys, zs)
            mx = dx / dist_3d
            my = dy / dist_3d
            mz = dz / dist_3d
            
            # Optionnel : Forcer une aimantation saturée loin du point de Bloch
            # pour simuler un état micromagnétique plus réaliste
            # (Ici, le vecteur est déjà unitaire de base, ce qui est parfait pour tester le code)
            
        magnetization[i] = [mx, my, mz]
        
    # 4. Sauvegarder dans sol.in
    np.savetxt(sol_path, magnetization, delimiter='\t', fmt='%.8e')
    print(f"\nSuccès ! Fichier '{sol_filename}' (Point de Bloch de Surface) généré à l'adresse :\n{sol_path}")

if __name__ == '__main__':
    generate_surface_bloch_point_sol("cube.msh", "sol.in")
