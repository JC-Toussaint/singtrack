import os
import numpy as np
import meshio

def generate_vortex_sol(msh_filename, sol_filename):
    # 1. Gestion des chemins absolus avec le module 'os'
    # Récupère le dossier où se trouve le script Python actuel
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construit les chemins absolus complets vers les fichiers msh et sol
    msh_path = os.path.join(script_dir, msh_filename)
    sol_path = os.path.join(script_dir, sol_filename)
    
    print(f"Recherche du maillage à l'adresse : {msh_path}")
    
    # Double vérification de sécurité
    if not os.path.exists(msh_path):
        raise FileNotFoundError(
            f"Le fichier '{msh_filename}' est introuvable dans le dossier du script.\n"
            f"Dossier inspecté : {script_dir}\n"
            f"Vérifie que le fichier s'appelle bien ainsi et qu'il n'est pas resté dans un sous-dossier."
        )

    # 2. Lire le maillage
    mesh = meshio.read(msh_path)
    nodes = mesh.points  # Tableau (Nnodes, 3) -> x, y, z
    
    # 3. Initialiser la matrice d'aimantation (Nnodes, 3)
    magnetization = np.zeros_like(nodes)
    R_core = 10  # 10 nm
    
    for i, (x, y, z) in enumerate(nodes):
        mx = -y
        my =  x
        mz = 0
            
        magnetization[i] = [mx, my, mz]
       
    # 4. Sauvegarder dans sol.in
    with open(sol_path, "w") as f:
    # Première ligne
        f.write("#time : 0\n")

    # Une ligne par nœud : indice (base 0), mx, my, mz
        for i, (r, m) in enumerate(zip(nodes, magnetization)):
            x, y, z = r
            mx, my, mz = m
            f.write(f"{i}\t{x:.8e}\t{y:.8e}\t{z:.8e}\t{mx:.8e}\t{my:.8e}\t{mz:.8e}\n")

    print(f"\nSuccès ! Fichier '{sol_filename}' généré à l'adresse :\n{sol_path}")

if __name__ == '__main__':
    generate_vortex_sol("cube.msh", "sim_iter0.sol")
