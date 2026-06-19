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
    for i, (x, y, z) in enumerate(nodes):
        mx = +y-x
        my = -x-y
        mz = -z 
            
        magnetization[i] = [mx, my, mz]
        
    # 4. Sauvegarder dans sol.in
    np.savetxt(sol_path, magnetization, delimiter='\t', fmt='%.8e')
    print(f"\nSuccès ! Fichier '{sol_filename}' généré à l'adresse :\n{sol_path}")

if __name__ == '__main__':
    generate_vortex_sol("cube.msh", "sol.in")
