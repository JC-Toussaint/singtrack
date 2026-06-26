import os
import re
import time
import math
import cmath
from pathlib import Path
import numpy as np
import meshio

# --- CONFIGURATION & PARAMÈTRES ---
DIRECTORY_PATH = Path(".")  # Dossier où se trouvent les fichiers *.sol ('.' signifie dossier courant)
# TOL : Seuil numérique de tolérance pour éviter les divisions par zéro lors du calcul de déterminants, 
# ou pour identifier des valeurs considérées comme purement réelles/imaginaires (proches de la précision machine).
TOL = 1e-16

# --- CLASSES DE RÉSULTATS ---
# Remplacent les structures C++ pour l'analyse interne (Volume) et de surface (2D)
class BlochPointResult:
    def __init__(self, iter_num, time_val, pos, curl, eigvals, bp_type):
        self.iter = iter_num
        self.time = time_val
        self.pos = pos        # np.array([x, y, z])
        self.curl = curl      # np.array([cx, cy, cz])
        self.eigvals = eigvals # np.array([e0, e1, e2]) complexes
        self.type = bp_type

class SurfaceSingularityResult:
    def __init__(self, iter_num, time_val, pos, curl_n, eigvals, surf_type, polarity, topological_charge):
        self.iter = iter_num
        self.time = time_val
        self.pos = pos        # np.array([x, y, z])
        self.curl_n = curl_n
        self.eigvals = eigvals # np.array([e0, e1]) complexes
        self.type = surf_type
        self.polarity = polarity
        self.topological_charge = topological_charge


# --- FONCTION 1 : LECTURE D'UN FICHIER SOLUTION ---
def load_magnetization(sol_filename):
    out_time = 0.0
    time_found = False
    mag_list = []

    if not os.path.exists(sol_filename):
        raise FileNotFoundError(f"Fichier SOL manquant : {sol_filename}")

    # Parcours ligne par ligne du fichier contenant les en-têtes puis les données nodales
    with open(sol_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Traitement des lignes de commentaires / en-têtes (commençant par '#')
            if line.startswith('#'):
                # CORRECTION STRUCTURELLE : Recherche stricte en début de ligne de "## time:"
                if not time_found and line.startswith("## time:"):
                    try:
                        # Extraction et conversion de la notation scientifique en réel double précision
                        out_time = float(line.split(':')[1].strip())
                        time_found = True
                    except Exception:
                        print(f"Attention : Impossible de parser la valeur de temps dans {sol_filename}. Fixé à 0.0.")
                continue

            # Lecture des données numériques nodales
            columns = [float(x) for x in line.split()]
            
            # Format feeLLGood : idx(colonne 0) mx(colonne 1) my(colonne 2) mz(colonne 3) ...
            if len(columns) < 4:
                raise ValueError(f"Ligne incomplète ou mal formée dans {sol_filename} (Moins de 4 colonnes trouvées).")

            # Récupération des composantes du vecteur d'aimantation (colonnes 1, 2 et 3)
            mag_list.append([columns[1], columns[2], columns[3]])

    if not time_found:
        print(f"Attention : En-tête '## time:' absent dans {sol_filename}. Fixé à 0.0.")

    return np.array(mag_list), out_time


# --- FONCTION 2 : LECTURE DU MAILLAGE VIA MESHIO (AVEC SQUIZZAGE DU HOR-FORMAT) ---
def load_mesh_gmsh(msh_filename, expected_nodes):
    # 1. Pré-traitement et nettoyage du fichier maillage
    clean_filename = msh_filename + ".clean_tmp"
    inside_corrupted_zone = False
    skipped_lines_count = 0

    with open(msh_filename, 'r') as src_file, open(clean_filename, 'w') as dst_file:
        for line in src_file:
            if "$EndMeshFormat" in line:
                dst_file.write(line)
                inside_corrupted_zone = True # On entre dans la zone tampon potentiellement corrompue
                continue

            if "$Nodes" in line:
                inside_corrupted_zone = False # Fin de la zone de lignes parasites

            if inside_corrupted_zone:
                skipped_lines_count += 1
                continue # Ignore la ligne parasite

            dst_file.write(line)

    if skipped_lines_count > 0:
        print(f"\n[ATTENTION] Le fichier maillage {msh_filename} contenait {skipped_lines_count} ligne(s) non standard.")
        print("            Ces lignes ont ete squizees pour assurer une lecture correcte.\n")

    # 2. Chargement des données géométriques via meshio
    try:
        mesh_data = meshio.read(clean_filename)
    finally:
        if os.path.exists(clean_filename):
            os.remove(clean_filename) # Sécurité : suppression du fichier temporaire

    points = mesh_data.points  # Matrice (N, 3) des coordonnées X, Y, Z
    tetrahedrons = []
    triangles = []

    # Extraction de la topologie de connectivité des éléments
    for cell in mesh_data.cells:
        if cell.type == "triangle":
            triangles.extend(cell.data)
        elif cell.type == "tetra":
            tetrahedrons.extend(cell.data)

    tetrahedrons = np.array(tetrahedrons, dtype=int)
    triangles = np.array(triangles, dtype=int)

    if len(tetrahedrons) == 0:
        raise ValueError("Aucun élément tétraédrique trouvé.")

    return points, tetrahedrons, triangles


# --- RÉORIENTATION DES TRIANGLES ---
def orient_triangles_outward(points, tetrahedrons, triangles):
    if len(triangles) == 0:
        return triangles

    print("Structure de recherche des tétraèdres en cours... ")
    # Construction d'une table d'adjacence inverse (Nœud -> Liste des tétraèdres adjacents)
    node_to_tetra = {i: [] for i in range(len(points))}
    for t_idx, tetra in enumerate(tetrahedrons):
        for node in tetra:
            node_to_tetra[node].append(t_idx)

    print("Correction de l'orientation des triangles...")
    oriented_triangles = np.copy(triangles)
    corrected_count = 0

    for i, tri in enumerate(oriented_triangles):
        # Intersection des listes de tétraèdres pour trouver celui contenant les 3 nœuds du triangle
        t0_set = set(node_to_tetra[tri[0]])
        t1_set = set(node_to_tetra[tri[1]])
        t2_set = set(node_to_tetra[tri[2]])
        
        final_intersect = t0_set.intersection(t1_set).intersection(t2_set)
        if not final_intersect:
            continue

        tetra_idx = list(final_intersect)[0]
        tetra = tetrahedrons[tetra_idx]
        
        # Identification du 4ème nœud du tétraèdre (interne)
        internal_node = [n for n in tetra if n not in tri][0]

        p0, p1, p2 = points[tri[0]], points[tri[1]], points[tri[2]]
        p_in = points[internal_node]

        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        v_to_internal = p_in - p0

        # Si le produit scalaire est positif, la normale pointe vers l'intérieur
        if np.dot(normal, v_to_internal) > 0:
            # Permutation de deux sommets pour inverser le produit vectoriel
            oriented_triangles[i][0], oriented_triangles[i][1] = oriented_triangles[i][1], oriented_triangles[i][0]
            corrected_count += 1

    print(f"Orientation terminée. {corrected_count} / {len(triangles)} triangles inversés.")
    return oriented_triangles


# --- ANALYSE POINT DE BLOCH (VOLUME) ---
def analyze_bloch_point(current_iter, current_time, nodes_coords, nodes_mag):
    # A_m : Espace des aimantations relativement au premier nœud
    A_m = np.zeros((3, 3))
    A_m[:, 0] = nodes_mag[:, 1] - nodes_mag[:, 0]
    A_m[:, 1] = nodes_mag[:, 2] - nodes_mag[:, 0]
    A_m[:, 2] = nodes_mag[:, 3] - nodes_mag[:, 0]

    if abs(np.linalg.det(A_m)) < TOL:
        return None

    # Résolution par moindres carrés / QR pour trouver les coordonnées barycentriques locales (u, v, w)
    vec_local = np.linalg.solve(A_m, -nodes_mag[:, 0])

    # Vérification stricte d'inclusion (u>0, v>0, w>0 et u+v+w<1)
    if np.all(vec_local > 0) and np.all(vec_local < 1) and np.sum(vec_local) < 1:
        B_geo = np.zeros((3, 3))
        B_geo[:, 0] = nodes_coords[:, 1] - nodes_coords[:, 0]
        B_geo[:, 1] = nodes_coords[:, 2] - nodes_coords[:, 0]
        B_geo[:, 2] = nodes_coords[:, 3] - nodes_coords[:, 0]

        sol_cartesian = nodes_coords[:, 0] + np.dot(B_geo, vec_local)

        # Calcul des gradients spatiaux (Jacobien)
        A_space = np.ones((4, 4))
        A_space[:, 1:] = nodes_coords.T

        if abs(np.linalg.det(A_space)) < TOL:
            return None

        inv_A_space = np.linalg.inv(A_space)
        coeff_x = np.dot(inv_A_space, nodes_mag[0, :].T)
        coeff_y = np.dot(inv_A_space, nodes_mag[1, :].T)
        coeff_z = np.dot(inv_A_space, nodes_mag[2, :].T)

        jac = np.zeros((3, 3))
        jac[0, :] = coeff_x[1:]
        jac[1, :] = coeff_y[1:]
        jac[2, :] = coeff_z[1:]

        # Calcul analytique du Rotationnel
        curl_x = coeff_z[2] - coeff_y[3]
        curl_y = coeff_x[3] - coeff_z[1]
        curl_z = coeff_y[1] - coeff_x[2]
        curl = np.array([curl_x, curl_y, curl_z])

        # Analyse spectrale
        eigvals = np.linalg.eigvals(jac)
        all_real = np.all(np.abs(eigvals.imag) < TOL)

        bp_type = "unknown"
        if all_real:
            r = eigvals.real
            if np.all(r > 0): bp_type = "source"
            elif np.all(r < 0): bp_type = "sink"
            else:
                bp_type = "saddle_2in-1out" if (r[0]*r[1]*r[2] > 0) else "saddle_1in-2out"
        else:
            # Identification de la valeur propre réelle isolée
            real_idx = np.argmin(np.abs(eigvals.imag))
            lamb1 = eigvals[real_idx].real
            comp_idx = [i for i in range(3) if i != real_idx][0]
            a = eigvals[comp_idx].real

            if lamb1 > 0 and a > 0: bp_type = "spiral_source"
            elif lamb1 > 0 and a < 0: bp_type = "spiral_saddle_2in-1out_tail_to_tail"
            elif lamb1 < 0 and a > 0: bp_type = "spiral_saddle_1in-2out_head_to_head"
            elif lamb1 < 0 and a < 0: bp_type = "spiral_sink"

        return BlochPointResult(current_iter, current_time, sol_cartesian, curl, eigvals, bp_type)
    
    return None


# --- ANALYSE SINGULARITÉ DE SURFACE ---
def analyze_surface_singularity(current_iter, current_time, ns_coords, ns_mag):
    v1 = ns_coords[:, 1] - ns_coords[:, 0]
    v2 = ns_coords[:, 2] - ns_coords[:, 0]
    n = np.cross(v1, v2)
    n_norm = np.linalg.norm(n)
    
    if n_norm == 0:
        return None
    n /= n_norm

    triangle_area = 0.5 * n_norm

    # Repère local orthonormé
    t0 = v1 / np.linalg.norm(v1)
    t1 = np.cross(n, t0)

    # Projections
    m_t0 = np.dot(ns_mag.T, t0)
    m_t1 = np.dot(ns_mag.T, t1)
    m_n  = np.dot(ns_mag.T, n)

    c_t0 = np.dot(ns_coords.T, t0)
    c_t1 = np.dot(ns_coords.T, t1)

    A_surf = np.zeros((2, 2))
    A_surf[0, 0] = m_t0[1] - m_t0[0]
    A_surf[0, 1] = m_t1[1] - m_t1[0]
    A_surf[1, 0] = m_t0[2] - m_t0[0]
    A_surf[1, 1] = m_t1[2] - m_t1[0]

    if abs(np.linalg.det(A_surf)) < TOL:
        return None

    vec_local = np.linalg.solve(A_surf, np.array([-m_t0[0], -m_t1[0]]))

    if vec_local[0] > 0 and vec_local[1] > 0 and (vec_local[0] + vec_local[1] < 1):
        sol_cartesian = ns_coords[:, 0] + vec_local[0] * v1 + vec_local[1] * v2
        
        p_val = m_n[0] + vec_local[0] * (m_n[1] - m_n[0]) + vec_local[1] * (m_n[2] - m_n[0])
        polarity = 1.0 if p_val > 0.0 else (-1.0 if p_val < 0.0 else 0.0)

        M_matrix = np.ones((3, 3))
        M_matrix[:, 1] = c_t0
        M_matrix[:, 2] = c_t1

        if abs(np.linalg.det(M_matrix)) < TOL:
            return None
        inv_M = np.linalg.inv(M_matrix)

        coeff_t0 = np.dot(inv_M, m_t0)
        coeff_t1 = np.dot(inv_M, m_t1)

        jac_2d = np.zeros((2, 2))
        jac_2d[0, 0] = coeff_t0[1]; jac_2d[0, 1] = coeff_t0[2]
        jac_2d[1, 0] = coeff_t1[1]; jac_2d[1, 1] = coeff_t1[2]

        eigvals = np.linalg.eigvals(jac_2d)
        curl_n = coeff_t1[1] - coeff_t0[2]

        is_complex = np.any(np.abs(eigvals.imag) > TOL)

        if is_complex:
            surf_type = "spiral_source" if (eigvals[0].real > 0) else "spiral_sink"
        else:
            r = eigvals.real
            if (r[0] > 0 and r[1] < 0) or (r[0] < 0 and r[1] > 0):
                surf_type = "saddle"
            else:
                surf_type = "source" if (r[0] > 0) else "sink"

        # --- CALCUL NUMÉRIQUE DE LA CHARGE TOPOLOGIQUE Q ---
        dm_dt0 = np.array([coeff_t0[1], coeff_t1[1], np.dot(inv_M[1, :], m_n)])
        dm_dt1 = np.array([coeff_t0[2], coeff_t1[2], np.dot(inv_M[2, :], m_n)])
        
        m_center = np.mean(ns_mag, axis=1)
        charge_density = np.dot(m_center, np.cross(dm_dt0, dm_dt1))
        topological_charge = (charge_density * triangle_area) / (4.0 * math.pi)

        return SurfaceSingularityResult(current_iter, current_time, sol_cartesian, curl_n, eigvals, surf_type, polarity, topological_charge)
    
    return None


# --- MAIN ---
def main():
    import sys
    if len(sys.argv) < 2:
        print("Erreur : Nom du fichier maillage (.msh) manquant.")
        print(f"Usage   : python {sys.argv[0]} <chemin_du_fichier_maillage.msh>")
        return 1
    msh_file = sys.argv[1]

    # 1. Lister tous les fichiers sol et extraire l'indice d'itération via Regex
    sol_files = []
    print(f"Recherche des fichiers solutions dans '{DIRECTORY_PATH}'...")
    if not DIRECTORY_PATH.exists():
        print(f"Dossier '{DIRECTORY_PATH}' inexistant.")
        return 1

    re_pattern = re.compile(R".*_iter(\d+)\.sol")
    for file in DIRECTORY_PATH.glob("*.sol"):
        filename = file.name
        match = re_pattern.match(filename)
        if match:
            num = int(match.group(1))
        else:
            num = 0
        sol_files.append((num, str(file)))

    if not sol_files:
        print("Aucun fichier solution (*.sol) trouvé.")
        return 1

    # Tri par itération chronologique
    sol_files.sort(key=lambda x: x[0])
    print(f"{len(sol_files)} fichiers solutions trouvés et triés.")

    # 2. Pré-chargement de la première solution pour la taille
    try:
        initial_mag, _ = load_magnetization(sol_files[0][1])
    except Exception as e:
        print(f"Erreur lors du pré-chargement : {e}")
        return 1

    # 3. CHARGEMENT ET CORRECTION DU MAILLAGE
    mesh_start = time.time()
    print(f"Chargement unique du maillage : {msh_file}...")
    try:
        points, tetrahedrons, triangles = load_mesh_gmsh(msh_file, initial_mag.shape[0])
    except Exception as e:
        print(f"Erreur de maillage : {e}")
        return 1
    
    triangles = orient_triangles_outward(points, tetrahedrons, triangles)
    mesh_duration = time.time() - mesh_start
    print(f"=> Temps de traitement du maillage : {mesh_duration:.4f} secondes.")

    global_bloch_points = []
    global_surface_singularities = []

    # 4. BOUCLE PRINCIPALE SUR TOUTES LES SOLUTIONS
    sol_start = time.time()
    for iter_num, file_path in sol_files:
        try:
            mag, current_time = load_magnetization(file_path)
        except Exception as e:
            print(f"Échec de lecture pour {file_path} ({e}), ignoré.")
            continue

        print("\n---------------------------------------------")
        print(f"Traitement de : {file_path} (Iteration: {iter_num} | Time: {current_time})...")

        if mag.shape[0] != points.shape[0]:
            print(f"Incohérence de taille de nœuds dans {file_path}, ignoré.")
            continue

        # --- ANALYSE VOLUME ---
        for nodes_idx in tetrahedrons:
            t_coords = points[nodes_idx].T  # Format (3, 4)
            t_mag = mag[nodes_idx].T        # Format (3, 4)

            # Test de pré-filtrage topologique rapide (sign_check)
            def sign_check(m, axis):
                has_pos = np.any(m[axis, :] > 0)
                has_neg = np.any(m[axis, :] < 0)
                return has_pos and has_neg

            if sign_check(t_mag, 0) and sign_check(t_mag, 1) and sign_check(t_mag, 2):
                res_bp = analyze_bloch_point(iter_num, current_time, t_coords, t_mag)
                if res_bp:
                    print(f" -> [BP Volume] détecté à : [{res_bp.pos[0]:.5f}, {res_bp.pos[1]:.5f}, {res_bp.pos[2]:.5f}] nm ({res_bp.type})")
                    global_bloch_points.append(res_bp)

        # --- ANALYSE SURFACE ---
        for nodes_idx in triangles:
            s_coords = points[nodes_idx].T  # Format (3, 3)
            s_mag = mag[nodes_idx].T        # Format (3, 3)

            res_surf = analyze_surface_singularity(iter_num, current_time, s_coords, s_mag)
            if res_surf:
                print(f" -> [Singularité Surface] ({res_surf.type}) à : [{res_surf.pos[0]:.5f}, {res_surf.pos[1]:.5f}, {res_surf.pos[2]:.5f}] nm "
                      f"| Polarité: {res_surf.polarity:.1f} | Charge Q: {res_surf.topological_charge:.4f}")
                global_surface_singularities.append(res_surf)

    sol_duration = time.time() - sol_start
    print(f"\n=> Temps total de traitement de tous les fichiers .sol : {sol_duration:.4f} secondes.")

    # 5. SAUVEGARDE GLOBALE COMPLETE
    print("\n-------------------------------------\nExécution de l'export final...")
    
    with open("all_volume_bloch_points.txt", "w") as f_vol:
        f_vol.write(f"{'#iter':<8}{'time':<15}{'x':>10}{'y':>10}{'z':>10}{'curl_x':>10}{'curl_y':>10}{'curl_z':>10}"
                    f"{'eig0_re':>12}{'eig0_im':>12}{'eig1_re':>12}{'eig1_im':>12}{'eig2_re':>12}{'eig2_im':>12}  type\n")
        for bp in global_bloch_points:
            f_vol.write(f"{bp.iter:<8}{bp.time:<15.6e}"
                        f"{bp.pos[0]:>10.4f}{bp.pos[1]:>10.4f}{bp.pos[2]:>10.4f}"
                        f"{bp.curl[0]:>10.4f}{bp.curl[1]:>10.4f}{bp.curl[2]:>10.4f}"
                        f"{bp.eigvals[0].real:>12.4f}{bp.eigvals[0].imag:>12.4f}"
                        f"{bp.eigvals[1].real:>12.4f}{bp.eigvals[1].imag:>12.4f}"
                        f"{bp.eigvals[2].real:>12.4f}{bp.eigvals[2].imag:>12.4f}  {bp.type}\n")

    with open("all_surface_singularities.txt", "w") as f_surf:
        f_surf.write(f"{'#iter':<8}{'time':<15}{'x':>10}{'y':>10}{'z':>10}{'curl_n':>12}{'polarity':>10}{'charge_Q':>12}"
                    f"{'eig0_re':>12}{'eig0_im':>12}{'eig1_re':>12}{'eig1_im':>12}  type\n")
        for ss in global_surface_singularities:
            f_surf.write(f"{ss.iter:<8}{ss.time:<15.6e}"
                        f"{ss.pos[0]:>10.4f}{ss.pos[1]:>10.4f}{ss.pos[2]:>10.4f}"
                        f"{ss.curl_n:>12.4f}{ss.polarity:>10.2f}{ss.topological_charge:>12.4f}"
                        f"{ss.eigvals[0].real:>12.4f}{ss.eigvals[0].imag:>12.4f}"
                        f"{ss.eigvals[1].real:>12.4f}{ss.eigvals[1].imag:>12.4f}  {ss.type}\n")

    print("Analyses multi-fichiers et exports complets terminés.")
    print("\n===== BILAN DES TEMPS DE CALCUL =====")
    print(f"Lecture & correction maillage : {mesh_duration:.4f} s")
    print(f"Traitement des fichiers .sol   : {sol_duration:.4f} s")
    print(f"Temps calcul total d'analyse  : {(mesh_duration + sol_duration):.4f} s")
    print("=====================================\n")
    return 0

if __name__ == "__main__":
    main()