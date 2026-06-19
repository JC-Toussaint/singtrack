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
MSH_FILE = "cube.msh"  # Chemin vers ton fichier GMSH
SOL_FILE = "sol.in"    # Chemin vers ton fichier d'aimantation
TIME_NS = 0.0          # Pas de temps
TOL = 1e-16

def load_mesh_and_magnetization(msh_filename, sol_filename):
    """
    Charge le maillage GMSH (tétraèdres + triangles de surface) et associe l'aimantation.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    msh_path = os.path.join(script_dir, msh_filename)
    sol_path = os.path.join(script_dir, sol_filename)

    if not os.path.exists(msh_path) or not os.path.exists(sol_path):
        raise FileNotFoundError("Fichier MSH ou SOL manquant.")

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

    # Lecture de l'aimantation
    mag = np.loadtxt(sol_path)
    if len(points) != len(mag):
        raise ValueError(f"Incohérence : {len(points)} nœuds vs {len(mag)} lignes de magnétisation.")

    return points, mag, tetrahedrons, triangles

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

def analyze_bloch_point(nodes_coords, nodes_mag):
    """
    Détecte un Point de Bloch (Volume) : m_x = m_y = m_z = 0 à l'intérieur du tétraèdre.
    """
    A_m = np.array([
        nodes_mag[1] - nodes_mag[0],
        nodes_mag[2] - nodes_mag[0],
        nodes_mag[3] - nodes_mag[0]
    ]).T

    if np.abs(np.linalg.det(A_m)) < TOL:
        return None

    vec_local = -1.0 * np.linalg.solve(A_m, nodes_mag[0])

    if (np.all(vec_local > 0) and np.all(vec_local < 1) and np.sum(vec_local) < 1):
        B_geo = np.array([
            nodes_coords[1] - nodes_coords[0],
            nodes_coords[2] - nodes_coords[0],
            nodes_coords[3] - nodes_coords[0]
        ]).T
        
        sol_cartesian = nodes_coords[0] + B_geo.dot(vec_local)
        
        # Jacobien spatial
        A_space = np.hstack([np.ones((4, 1)), nodes_coords])
        if np.abs(np.linalg.det(A_space)) < TOL: return None
        
        inv_A_space = np.linalg.inv(A_space)
        coeff_x = inv_A_space.dot(nodes_mag[:, 0])
        coeff_y = inv_A_space.dot(nodes_mag[:, 1])
        coeff_z = inv_A_space.dot(nodes_mag[:, 2])

        jac = np.array([coeff_x[1:], coeff_y[1:], coeff_z[1:]])

        curl_x = coeff_y[3] - coeff_z[2]
        curl_y = coeff_z[1] - coeff_x[3]
        curl_z = coeff_x[2] - coeff_y[1]

        eigvals = np.linalg.eigvals(jac)
        print(f'Eigenvalues : {eigvals}')

        # Classification du point de Bloch
        bp_type = "unknown"
        if np.all(np.isreal(eigvals)):
            if np.all(eigvals)>0:
                bp_type = "source" if eigvals[0] > 0 else "sink"
            else:
                bp_type = "saddle 2 in - 1 out" if np.prod(eigval) > 0 else "saddle 1 in - 2 out"
        else:
            # Recherche de la valeur propre réelle (lamb1) et de la partie réelle de la complexe (lamb2)
            real_mask = np.isreal(eigvals) # masque pour selectionner les valeurs propres réelles
            lamb1 = np.real(eigvals[real_mask])[0] # premiere valeur propre reelle
            a    = np.real(eigvals[~real_mask])[0] # partie reelle de la premiere valeur propre complex
            
            if lamb1 > 0 and a > 0: bp_type = "spiral source"
            elif lamb1 > 0 and a < 0: bp_type = "spiral saddle 2 in - 1out tail to tail"
            elif lamb1 < 0 and a > 0: bp_type = "spiral saddle 1 in - 2out head to head"
            elif lamb1 < 0 and a < 0: bp_type = "spiral sink"

        return [TIME_NS, sol_cartesian[0], sol_cartesian[1], sol_cartesian[2], 
                curl_x, curl_y, curl_z, eigvals[0], eigvals[1], eigvals[2], bp_type]
    return None

def analyze_surface_singularity(ns_coords, ns_mag):
    """
    Analyse une facette triangulaire de la SURFACE pour y chercher :
    - Un Point de Bloch de surface (m_x = m_y = m_z = 0 localement)
    - Un Vortex / Antivortex de surface (m_in_plane = 0, avec Polarité hors-plan)
    """
    # 1. Définition de la base locale de la facette (t0, t1: plan, n: normale sortante)
    v1 = ns_coords[1] - ns_coords[0]
    v2 = ns_coords[2] - ns_coords[0]
    n = np.cross(v1, v2)
    n_norm = np.linalg.norm(n)
    if n_norm == 0: return None
    n = n / n_norm
    
    t0 = v1 / np.linalg.norm(v1)
    t1 = np.cross(n, t0)

    # 2. Projection de l'aimantation dans la base locale
    m_t0 = ns_mag.dot(t0)
    m_t1 = ns_mag.dot(t1)
    m_n  = ns_mag.dot(n)

    # Coordonnées 2D projetées sur la facette
    c_t0 = ns_coords.dot(t0)
    c_t1 = ns_coords.dot(t1)

    # 3. Résolution de l'annulation des composantes planaires (m_t0 = 0, m_t1 = 0)
    A_surf = np.array([
        [m_t0[1] - m_t0[0], m_t0[2] - m_t0[0]],
        [m_t1[1] - m_t1[0], m_t1[2] - m_t1[0]]
    ]).T

    if np.abs(np.linalg.det(A_surf)) < TOL:
        return None

    vec_local = -1.0 * np.linalg.solve(A_surf, np.array([m_t0[0], m_t1[0]]))

    # Vérification si la singularité est dans le triangle
    if (vec_local[0] > 0 and vec_local[1] > 0 and (vec_local[0] + vec_local[1] < 1)):
        # Position cartésienne exacte
        sol_cartesian = ns_coords[0] + vec_local[0]*v1 + vec_local[1]*v2
        
        # Calcul de la polarité au point (composante normale m_n)
        # Interpolation linéaire de m_n au point de la singularité
        p_val = m_n[0] + vec_local[0]*(m_n[1] - m_n[0]) + vec_local[1]*(m_n[2] - m_n[0])
        polarity = np.sign(p_val)

        # Interpolation du Jacobien 2D pour la classification (Vortex vs Antivortex)
        M_matrix = np.hstack([np.ones((3, 1)), c_t0.reshape(-1,1), c_t1.reshape(-1,1)])
        if np.abs(np.linalg.det(M_matrix)) < TOL: return None
        inv_M = np.linalg.inv(M_matrix)
        
        coeff_t0 = inv_M.dot(m_t0)  # [constante, d(mt0)/dt0, d(mt0)/dt1]
        coeff_t1 = inv_M.dot(m_t1)  # [constante, d(mt1)/dt0, d(mt1)/dt1]
        jac_2d = np.array([coeff_t0[1:], coeff_t1[1:]])
        eigvals = np.linalg.eigvals(jac_2d)

        # Composante normale du rotationnel : d(mt1)/dt0 - d(mt0)/dt1
        curl_n = coeff_t1[1] - coeff_t0[2]

        # 1. On regarde d'abord la structure géométrique du champ planaire (Jacobien)
        if not np.all(np.isreal(eigvals)):
            # Valeurs propres complexes = rotation évidente
            surf_type = "spiral source" if np.real(eigvals[0])>0 else "spiral sink"
        else:
            # Valeurs propres réelles
            if np.sign(eigvals[0]) != np.sign(eigvals[1]):
                # Signes opposés = Point selle
                surf_type = "saddle"
            else:
                surf_type = "source" if np.all(eigvals)>0 else "sink"

        return [TIME_NS, sol_cartesian[0], sol_cartesian[1], sol_cartesian[2], curl_n, eigvals[0], eigvals[1], surf_type, polarity]
    
    return None


# --- SCRIPT PRINCIPAL ---
def main():
    print(f"Chargement de {MSH_FILE} et {SOL_FILE}...")
    points, mag, tetrahedrons, triangles = load_mesh_and_magnetization(MSH_FILE, SOL_FILE)
    
    # RE-ORIENTATION DES TRIANGLES
    triangles = orient_triangles_outward(points, tetrahedrons, triangles)

    bloch_points_list = []
    volume_vortices_list = []
    surface_sings_list = []

    # 1. ANALYSE DU VOLUME (Tétraèdres)
    print(f"Analyse de {len(tetrahedrons)} tétraèdres (Volume)...")
    for nodes_idx in tetrahedrons:
        t_coords = points[nodes_idx]
        t_mag = mag[nodes_idx]

        # Points de Bloch en Volume
        if (np.unique(np.sign(t_mag[:, 0])).size > 1 and 
            np.unique(np.sign(t_mag[:, 1])).size > 1 and 
            np.unique(np.sign(t_mag[:, 2])).size > 1):
            res_bp = analyze_bloch_point(t_coords, t_mag)
            if res_bp:
                print(f" -> [BP Volume] détecté à : [{res_bp[1]:.5f}, {res_bp[2]:.5f}, {res_bp[3]:.5f}] nm ({res_bp[-1]})")
                bloch_points_list.append(res_bp)

    # 2. ANALYSE DE LA SURFACE (Triangles externes)
    if len(triangles) > 0:
        print(f"Analyse de {len(triangles)} triangles (Surface)...")
        for nodes_idx in triangles:
            s_coords = points[nodes_idx]
            s_mag = mag[nodes_idx]

            res_surf = analyze_surface_singularity(s_coords, s_mag)
            if res_surf:
                print(f" -> [Singularité Surface] ({res_surf[7]}) à : [{res_surf[1]:.5f}, {res_surf[2]:.5f}, {res_surf[3]:.5f}] nm | Polarité: {res_surf[8]}")
                surface_sings_list.append(res_surf)
    else:
        print("Attention : Aucun élément triangulaire de surface trouvé dans le fichier MSH.")

    # --- SAUVEGARDE DES RÉSULTATS ---
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    np.savetxt(script_dir / 'volume_bloch_points.txt', np.array(bloch_points_list, dtype=object), 
               delimiter='\t', fmt="%s", header='time(ns)\tx\ty\tz\ttype')
    np.savetxt(script_dir / 'surface_singularities.txt', np.array(surface_sings_list, dtype=object), 
               delimiter='\t', fmt="%s", header='time(ns)\tx\ty\tz\ttype\tpolarity')

    print('\nTraitements terminés. Fichiers exportés avec succès.')

if __name__ == '__main__':
    main()
