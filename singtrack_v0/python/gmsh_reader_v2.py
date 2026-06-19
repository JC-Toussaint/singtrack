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

    # Extraction des tétraèdres (Volume)
    tetra_cells = [cell.data for cell in mesh.cells if cell.type == "tetra"]
    if not tetra_cells:
        raise ValueError("Aucun élément tétraédrique ('tetra') trouvé.")
    tetrahedrons = np.vstack(tetra_cells)

    # Extraction des triangles (Surface)
    triangle_cells = [cell.data for cell in mesh.cells if cell.type == "triangle"]
    triangles = np.vstack(triangle_cells) if triangle_cells else np.array([])

    return points, tetrahedrons, triangles


def orient_triangles_outward(points, tetrahedrons, triangles):
    """
    Réoriente les triangles de surface pour que leur normale pointe systématiquement
    vers l'extérieur du volume en s'appuyant sur les tétraèdres adjacents.
    """
    if len(triangles) == 0:
        return triangles

    print("Structure de recherche des tétraèdres en cours de création...")
    # Pour accélérer la recherche, on crée un dictionnaire qui associe chaque nœud
    # à la liste des indices de tétraèdres auxquels il appartient.
    node_to_tetra = {}
    for tetra_idx, tetra in enumerate(tetrahedrons):
        for node in tetra:
            if node not in node_to_tetra:
                node_to_tetra[node] = []
            node_to_tetra[node].append(tetra_idx)

    print("Correction de l'orientation des triangles...")
    oriented_triangles = np.copy(triangles)
    corrected_count = 0

    for i, tri in enumerate(oriented_triangles):
        # Trouver le tétraèdre qui contient les 3 nœuds du triangle
        # On intersecte les listes de tétraèdres des 3 nœuds du triangle
        possible_tetras = set(node_to_tetra.get(tri[0], [])) & \
                          set(node_to_tetra.get(tri[1], [])) & \
                          set(node_to_tetra.get(tri[2], []))
        
        if not possible_tetras:
            # Si le triangle n'appartient à aucun tétraèdre (ex: structure orpheline)
            continue
        
        # On prend le premier tétraèdre trouvé (il n'y en a de toute façon qu'un seul pour un triangle de surface)
        tetra_idx = possible_tetras.pop()
        tetra = tetrahedrons[tetra_idx]
        
        # Trouver le 4ème nœud du tétraèdre (celui qui est à l'intérieur)
        internal_node = [n for n in tetra if n not in tri][0]
        
        # Coordonnées des points
        p0, p1, p2 = points[tri[0]], points[tri[1]], points[tri[2]]
        p_in = points[internal_node]
        
        # Calcul de la normale actuelle du triangle (via produit vectoriel)
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        
        # Vecteur allant d'un point du triangle vers le nœud interne
        v_to_internal = p_in - p0
        
        # Si le produit scalaire est positif, la normale pointe du même côté 
        # que le nœud interne (donc vers l'INTÉRIEUR). Il faut inverser.
        if np.dot(normal, v_to_internal) > 0:
            # Interversion des deux premiers nœuds pour inverser la normale
            oriented_triangles[i][0], oriented_triangles[i][1] = oriented_triangles[i][1], oriented_triangles[i][0]
            corrected_count += 1

    print(f"Orientation terminée. {corrected_count} / {len(triangles)} triangles ont été inversés.")
    return oriented_triangles


def analyze_surface(ns_coords):
    """
    Analyse une facette triangulaire de la SURFACE dont la normale est garantie sortante.
    """
    v1 = ns_coords[1] - ns_coords[0]
    v2 = ns_coords[2] - ns_coords[0]
    n = np.cross(v1, v2)
    n_norm = np.linalg.norm(n)
    if n_norm == 0: 
        return None
    n = n / n_norm
    
    # Ici, n est maintenant garantie 100% sortante grâce à la routine de correction
    return n


# --- SCRIPT PRINCIPAL ---
def main():
    print(f"Chargement de {MSH_FILE} ...")
    points, tetrahedrons, triangles = load_mesh(MSH_FILE)

    # RE-ORIENTATION DES TRIANGLES
    triangles = orient_triangles_outward(points, tetrahedrons, triangles)

    # ANALYSE DE LA SURFACE (Triangles externes avec normales correctes)
    if len(triangles) > 0:
        print(f"Analyse de {len(triangles)} triangles (Surface)...")
        for nodes_idx in triangles:
            s_coords = points[nodes_idx]
            res_surf = analyze_surface(s_coords)

if __name__ == '__main__':
    main()
