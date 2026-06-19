# -*- coding: utf-8 -*-
"""
Analyseur de singularités micromagnétiques (Points de Bloch & Vortex - Volume & Surface)
Adapté pour maillages GMSH (.msh) et fichiers d'aimantation (sol.in)
"""

import os
from pathlib import Path
import math
import numpy as np
import meshio

# --- CONFIGURATION & PARAMÈTRES ---
MSH_FILE = "cube14.msh"  # Chemin vers ton fichier GMSH

def load_mesh(msh_filename):
    """
    Charge le maillage GMSH (tétraèdres + triangles de surface) et associe l'aimantation.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    msh_path = os.path.join(script_dir, msh_filename)

    if not os.path.exists(msh_path):
        raise FileNotFoundError("Fichier MSH manquant.")

    mesh = meshio.read(msh_path)
    points = mesh.points  # (Nnodes, 3)
    print(mesh)

    # Extraction des tétraèdres (Volume)
    tetra_cells = [cell.data for cell in mesh.cells if cell.type == "tetra"]
    if not tetra_cells:
        raise ValueError("Aucun élément tétraédrique ('tetra') trouvé.")
    tetrahedrons = np.vstack(tetra_cells)

    # Extraction des triangles (Surface)
    triangle_cells = [cell.data for cell in mesh.cells if cell.type == "triangle"]
    triangles = np.vstack(triangle_cells) if triangle_cells else np.array([])

    return points, tetrahedrons, triangles

def analyze_surface(ns_coords):
    """
    Analyse une facette triangulaire de la SURFACE pour y chercher :
    """
    # 1. Définition de la base locale de la facette (t0, t1: plan, n: normale sortante)
    v1 = ns_coords[1] - ns_coords[0]
    v2 = ns_coords[2] - ns_coords[0]
    n = np.cross(v1, v2)
    n_norm = np.linalg.norm(n)
    if n_norm == 0: return None
    n = n / n_norm
    
    # S'assurer que la normale pointe vers l'extérieur (pour un cube centré en 0)
    if (np.dot(ns_coords[0], n)<0):
        print("Erreur")

    #barycenter = np.mean(ns_coords, axis=0)
    #if np.dot(barycenter, n) < 0: 
    #    n = -n

# --- SCRIPT PRINCIPAL ---
def main():
    print(f"Chargement de {MSH_FILE} ...")
    points, tetrahedrons, triangles = load_mesh(MSH_FILE)

    print(tetrahedrons)
    print(triangles)

    # ANALYSE DE LA SURFACE (Triangles externes)
    if len(triangles) > 0:
        print(f"Analyse de {len(triangles)} triangles (Surface)...")
        for nodes_idx in triangles:
            s_coords = points[nodes_idx]
            res_surf = analyze_surface(s_coords)

if __name__ == '__main__':
    main()
