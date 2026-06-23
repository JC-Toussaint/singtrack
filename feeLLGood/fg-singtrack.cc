#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <set>
#include <algorithm>
#include <memory>
#include <filesystem>
#include <stdexcept>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <regex>
#include <chrono>

// Inclusion de l'API Gmsh
#include <gmsh.h>

namespace fs = std::filesystem;
// Définition d'un type de matrice Eigen fixe (3 lignes, 4 colonnes) de doubles.
// Idéal pour stocker de manière contiguë les coordonnées (x,y,z) ou les vecteurs d'aimantation (mx,my,mz)
// aux 4 sommets d'un tétraèdre (volume).
using Matrix34d = Eigen::Matrix<double, 3, 4>;

// --- CONFIGURATION & PARAMÈTRES ---
const std::string DIRECTORY_PATH = "."; // Dossier où se trouvent les fichiers *.sol ('.' signifie dossier courant)
// TOL : Seuil numérique de tolérance pour éviter les divisions par zéro lors du calcul de déterminants, 
// ou pour identifier des valeurs considérées comme purement réelles/imaginaire (proches de la précision machine).
const double TOL = 1e-16;

// Structure représentant le maillage géométrique 3D non structuré
struct Mesh {
    Eigen::MatrixXd points;                  // Matrice dynamique (N lignes, 3 colonnes) stockant les coordonnées X, Y, Z de chaque nœud.
    std::vector<Eigen::Vector4i> tetrahedrons; // Liste d'éléments de volume : chaque Vector4i contient les 4 indices des nœuds formant un tétraèdre.
    std::vector<Eigen::Vector3i> triangles;    // Liste d'éléments de surface : chaque Vector3i contient les 3 indices des nœuds formant un triangle de peau.
};

// Structures de résultats complètes pour l'analyse topologique interne (Volume)
struct BlochPointResult {
    int iter;                 // Numéro de l'itération temporelle issue du fichier .sol
    double time;              // Temps physique associé à la simulation micromagnétique
    Eigen::Vector3d pos;      // Position spatiale exacte (x, y, z) du point de Bloch calculée par interpolation
    Eigen::Vector3d curl;     // Rotationnel de l'aimantation (rot m) évalué au centre/point singulier
    Eigen::Vector3cd eigvals; // Les 3 valeurs propres complexes du jacobien de l'aimantation (classification de la singularité)
    std::string type;         // Classification textuelle (ex: source, sink, spiral source, saddle...)
};

// Structures de résultats complètes pour l'analyse des défauts de surface (2D)
struct SurfaceSingularityResult {
    int iter;                 // Numéro de l'itération temporelle
    double time;              // Temps physique de la simulation
    Eigen::Vector3d pos;      // Position spatiale (x, y, z) de la singularité sur le triangle de surface
    double curl_n;            // Composante normale du rotationnel à la surface (curl m \cdot \vec{n})
    Eigen::Vector2cd eigvals; // Les 2 valeurs propres complexes issues du jacobien projeté sur le plan tangent de la surface
    std::string type;         // Classification textuelle bidimensionnelle (ex: source, sink, saddle, spiral...)
    double polarity;          // Polarité de la singularité de surface (-1 ou +1, signe de la composante normale de l'aimantation)
};

// --- FONCTION 1 : LECTURE D'UN FICHIER SOLUTION (ADAPTÉE POUR FEELLGOOD) ---
// Charge le champ d'aimantation normalisé \vec{m}(x,y,z) associé à chaque nœud du maillage pour un instant donné.
Eigen::MatrixXd load_magnetization(const std::string& sol_filename, double& out_time) {
    std::ifstream sol_file(sol_filename);
    if (!sol_file.is_open()) throw std::runtime_error("Fichier SOL manquant : " + sol_filename);
    
    std::string line;
    out_time = 0.0;
    bool time_found = false;
    std::vector<Eigen::Vector3d> mag_list;

    while (std::getline(sol_file, line)) {
        if (line.empty()) continue;

        // Traitement des lignes de commentaires / en-têtes (commençant par '#')
        if (line[0] == '#') {
            // CORRECTION : On vérifie que la ligne commence précisément par "## time:" 
            // pour éviter d'intercepter "## real-world time:" par erreur.
            if (!time_found && line.rfind("## time:", 0) == 0) {
                size_t pos_colon = line.find(':');
                if (pos_colon != std::string::npos) {
                    try {
                        out_time = std::stod(line.substr(pos_colon + 1));
                        time_found = true;
                    } catch (...) {
                        std::cerr << "Attention : Impossible de parser la valeur de temps dans " << sol_filename << ". Fixé à 0.0.\n";
                    }
                }
            }
            continue; // Passer à la ligne suivante (c'est une ligne d'en-tête)
        }
        
        // Lecture des données numériques nodales
        std::stringstream ss(line);
        double val;
        std::vector<double> columns;
        
        while (ss >> val) {
            columns.push_back(val);
        }
        
        // Structure du format feeLLGood : idx(colonne 0) mx(colonne 1) my(colonne 2) mz(colonne 3) ...
        if (columns.size() < 4) {
            throw std::runtime_error("Ligne incomplète dans " + sol_filename + " (Moins de 4 colonnes trouvées).");
        }
        
        // Récupération des composantes du vecteur d'aimantation (colonnes 1, 2 et 3)
        double mx = columns[1];
        double my = columns[2];
        double mz = columns[3];
        
        mag_list.push_back(Eigen::Vector3d(mx, my, mz));
    }
    sol_file.close();

    if (!time_found) {
        std::cerr << "Attention : En-tête '## time:' absent dans " << sol_filename << ". Fixé à 0.0.\n";
    }

    Eigen::MatrixXd mag(mag_list.size(), 3);
    for (size_t i = 0; i < mag_list.size(); ++i) {
        mag.row(i) = mag_list[i];
    }
    return mag;
}

// --- FONCTION 2 : LECTURE DU MAILLAGE VIA GMSH API (AVEC SQUIZZAGE DES LIGNES HORS-FORMAT) ---
// Cette routine effectue d'abord un nettoyage des sections non standard du fichier .msh
// avant d'appeler le moteur Gmsh pour parser les éléments géométriques (Triangles et Tétraèdres).
Mesh load_mesh_gmsh(const std::string& msh_filename, size_t expected_nodes) {
    Mesh mesh;

    // 1. Pré-traitement et nettoyage du fichier maillage
    // Certains générateurs ou convertisseurs écrivent des lignes de métadonnées parasites juste après le bloc
    // "$EndMeshFormat" et avant "$Nodes". Gmsh refuse d'ouvrir le fichier si ces lignes ne respectent pas sa grammaire.
    std::ifstream src_file(msh_filename);
    if (!src_file.is_open()) {
        throw std::runtime_error("Impossible d'ouvrir le fichier maillage : " + msh_filename);
    }

    std::string clean_filename = msh_filename + ".clean_tmp";
    std::ofstream dst_file(clean_filename);
    
    std::string line;
    bool inside_corrupted_zone = false;
    int skipped_lines_count = 0;

    while (std::getline(src_file, line)) {
        // Détection du tag de fin du format de maillage (v2 ou v4)
        if (line.find("$EndMeshFormat") != std::string::npos) {
            dst_file << line << "\n";
            inside_corrupted_zone = true; // On entre dans la zone tampon potentiellement corrompue
            continue;
        }

        // Détection du tag réglementaire suivant indiquant le début de la table des nœuds
        if (line.find("$Nodes") != std::string::npos) {
            inside_corrupted_zone = false; // Fin de la zone de lignes parasites, reprise du flux standard
        }

        // Si on se trouve dans l'interstice et que la ligne n'est pas le début du bloc $Nodes, on l'exclut du fichier propre
        if (inside_corrupted_zone) {
            skipped_lines_count++;
            continue; // Ignore la ligne parasite
        }

        // Écrit les lignes valides dans le fichier temporaire destiné à Gmsh
        dst_file << line << "\n";
    }
    src_file.close();
    dst_file.close();

    // Avertir l'utilisateur si des anomalies de formatage ont été détectées et nettoyées
    if (skipped_lines_count > 0) {
        std::cout << "\n[ATTENTION] Le fichier maillage " << msh_filename << " contenait " 
                  << skipped_lines_count << " ligne(s) non standard entre $EndMeshFormat et $Nodes.\n"
                  << "            Ces lignes ont ete squizees pour assurer une lecture correcte.\n\n";
    }

    // 2. Initialisation du moteur Gmsh API et ouverture du fichier nettoyé
    gmsh::initialize();
    gmsh::option::setNumber("General.Terminal", 0); // Désactive les logs verbeux de Gmsh sur la sortie standard
    
    try {
        gmsh::open(clean_filename);
    } catch (...) {
        std::filesystem::remove(clean_filename); // Sécurité : suppression du fichier temporaire en cas de crash
        gmsh::finalize();
        throw std::runtime_error("Impossible d'ouvrir le fichier maillage via Gmsh : " + msh_filename);
    }

    // Suppression immédiate du fichier temporaire, les données étant désormais chargées en mémoire par l'API Gmsh
    std::filesystem::remove(clean_filename);

    // 3. Extraction classique des nœuds et éléments via l'API Gmsh
    std::vector<size_t> nodeTags;
    std::vector<double> coord;
    std::vector<double> parametricCoord;
    // Récupère l'intégralité des nœuds du modèle Gmsh sous forme de tableaux plats à 1 dimension (coord = [x0, y0, z0, x1, y1, z1, ...])
    gmsh::model::mesh::getNodes(nodeTags, coord, parametricCoord);

    if (nodeTags.empty()) {
        gmsh::finalize();
        throw std::runtime_error("Le maillage chargé ne contient aucun nœud.");
    }

    // Dimensionnement de la matrice de points. Les lignes correspondent aux indices physiques (0 à expected_nodes - 1)
    mesh.points.resize(expected_nodes, 3);

    for (size_t i = 0; i < nodeTags.size(); ++i) {
        size_t tag = nodeTags[i];
        size_t idx = tag - 1; // Gmsh utilise une numérotation à base 1 pour les tags. On décale à base 0 pour l'indexation C++/Eigen.

        if (idx < expected_nodes) {
            mesh.points(idx, 0) = coord[3 * i];     // Coordonnée X
            mesh.points(idx, 1) = coord[3 * i + 1]; // Coordonnée Y
            mesh.points(idx, 2) = coord[3 * i + 2]; // Coordonnée Z
        }
    }

    // Extraction de la topologie de connectivité des éléments (Triangles / Tétraèdres)
    std::vector<int> elementTypes;
    std::vector<std::vector<size_t>> elementTags;
    std::vector<std::vector<size_t>> nodeTagsPerElement;
    gmsh::model::mesh::getElements(elementTypes, elementTags, nodeTagsPerElement);

    for (size_t i = 0; i < elementTypes.size(); ++i) {
        int type = elementTypes[i];
        const auto& nodes = nodeTagsPerElement[i];

        if (type == 2) { // Type 2 dans la nomenclature Gmsh correspond à un triangle linéaire à 3 nœuds (2D)
            for (size_t j = 0; j < nodes.size(); j += 3) {
                int idx0 = static_cast<int>(nodes[j] - 1);
                int idx1 = static_cast<int>(nodes[j+1] - 1);
                int idx2 = static_cast<int>(nodes[j+2] - 1);
                mesh.triangles.push_back(Eigen::Vector3i(idx0, idx1, idx2));
            }
        } 
        else if (type == 4) { // Type 4 dans la nomenclature Gmsh correspond à un tétraèdre linéaire à 4 nœuds (3D)
            for (size_t j = 0; j < nodes.size(); j += 4) {
                int idx0 = static_cast<int>(nodes[j] - 1);
                int idx1 = static_cast<int>(nodes[j+1] - 1);
                int idx2 = static_cast<int>(nodes[j+2] - 1);
                int idx3 = static_cast<int>(nodes[j+3] - 1);
                mesh.tetrahedrons.push_back(Eigen::Vector4i(idx0, idx1, idx2, idx3));
            }
        }
    }

    gmsh::finalize(); // Libération ordonnée de la mémoire interne allouée par la bibliothèque Gmsh

    if (mesh.tetrahedrons.empty()) throw std::runtime_error("Aucun élément tétraédrique trouvé.");
    return mesh;
}

// --- RÉORIENTATION DES TRIANGLES ---
// Force l'orientation des triangles de surface vers l'extérieur de la géométrie 3D (vecteur normal sortant).
// C'est une étape cruciale pour le calcul cohérent du signe de la polarité et du rotationnel normal en surface.
std::vector<Eigen::Vector3i> orient_triangles_outward(const Eigen::MatrixXd& points, 
                                                      const std::vector<Eigen::Vector4i>& tetrahedrons, 
                                                      const std::vector<Eigen::Vector3i>& triangles) {
    if (triangles.empty()) return triangles;
    std::cout << "Structure de recherche des tétraèdres en cours... " << std::endl;
    
    // Construction d'une table d'adjacence inverse (Nœud -> Liste des tétraèdres adjacents).
    // Permet d'éviter une recherche globale quadratique O(N_triangles * N_tétraèdres) très lourde.
    std::map<int, std::vector<int>> node_to_tetra;
    for (int t_idx = 0; t_idx < static_cast<int>(tetrahedrons.size()); ++t_idx) {
        for (int i = 0; i < 4; ++i) {
            node_to_tetra[tetrahedrons[t_idx][i]].push_back(t_idx);
        }
    }
    std::cout << "Correction de l'orientation des triangles..." << std::endl;
    
    //  Création d'une copie locale de la liste des triangles qui contiendra les indices corrigés (permutés si nécessaire)
    std::vector<Eigen::Vector3i> oriented_triangles = triangles;
    int corrected_count = 0; // Compteur du nombre de triangles dont l'ordre des sommets a été inversé
			     
    for (size_t i = 0; i < oriented_triangles.size(); ++i) {
        auto& tri = oriented_triangles[i];
        const auto& t0_list = node_to_tetra[tri[0]];  // Récupération des tétraèdres connectés au sommet 0
        const auto& t1_list = node_to_tetra[tri[1]];  // Récupération des tétraèdres connectés au sommet 1
        const auto& t2_list = node_to_tetra[tri[2]];  // Récupération des tétraèdres connectés au sommet 2
        std::set<int> intersect1, final_intersect;
        
        // Extraction des tétraèdres présents simultanément dans t0_list et t1_list (partagent l'arête [0-1])
        // Le std::inserter permet d'alimenter dynamiquement le conteneur ordonné 'intersect1'.
        std::set_intersection(t0_list.begin(), t0_list.end(), t1_list.begin(), t1_list.end(), std::inserter(intersect1, intersect1.begin()));

        // Finalisation de l'intersection avec la liste t2_list. Le résultat 'final_intersect' donne 
        // le ou les tétraèdres volumiques possédant la face complète (sommet 0, 1 et 2).
        // Pour un maillage de peau (manifold), un triangle de surface appartient exactement à UN SEUL tétraèdre interne.
        std::set_intersection(intersect1.begin(), intersect1.end(), t2_list.begin(), t2_list.end(), std::inserter(final_intersect, final_intersect.begin()));
        if (final_intersect.empty()) continue; // Si aucune intersection, le triangle est isolé ou orphelin.
        
        int tetra_idx = *final_intersect.begin(); // Récupère l'index du tétraèdre sous-jacent unique
        Eigen::Vector4i tetra = tetrahedrons[tetra_idx];
        int internal_node = -1;
        
        // Boucle pour identifier le 4ème nœud du tétraèdre, c'est-à-dire le nœud situé strictement *à l'intérieur* du volume.
        for (int n = 0; n < 4; ++n) {
            if (tetra[n] != tri[0] && tetra[n] != tri[1] && tetra[n] != tri[2]) {
                internal_node = tetra[n];
                break;
            }
        }
        
        // Extraction géométrique des coordonnées 3D des sommets pour calculs vectoriels
        Eigen::Vector3d p0 = points.row(tri[0]);
        Eigen::Vector3d p1 = points.row(tri[1]);
        Eigen::Vector3d p2 = points.row(tri[2]);
        Eigen::Vector3d p_in = points.row(internal_node); // Point interne de référence
        
        // Calcul des vecteurs directeurs de la face triangulaire
        Eigen::Vector3d v1 = p1 - p0;
        Eigen::Vector3d v2 = p2 - p0;
        // La règle de la main droite donne la normale : \vec{n} = \vec{v1} \times \vec{v2}
        Eigen::Vector3d normal = v1.cross(v2);
        // Vecteur pointant du sommet 0 vers le cœur du volume interne
        Eigen::Vector3d v_to_internal = p_in - p0;
        
        // Produit scalaire entre la normale calculée et le vecteur pointant vers l'intérieur.
        // S'il est positif, l'angle est aigu (< 90°), ce qui signifie que la normale pointe maladroitement VERS L'INTÉRIEUR.
        if (normal.dot(v_to_internal) > 0) {
            // Permutation de deux sommets (0 et 1). Cela inverse le sens du produit vectoriel (v1 cross v2), 
            // ce qui réoriente mécaniquement la normale vers l'extérieur.
            std::swap(tri[0], tri[1]);
            corrected_count++;
        }
    }
    std::cout << "Orientation terminée. " << corrected_count << " / " << triangles.size() << " triangles inversés.\n";
    return oriented_triangles;
}

// --- ANALYSE POINT DE BLOCH (VOLUME) ---
// Un point de Bloch est une singularité topologique micromagnétique ponctuelle où l'aimantation s'annule localement (\vec{m} = \vec{0}).
// On utilise une interpolation affine de l'aimantation à l'intérieur du tétraèdre pour trouver ce zéro et caractériser le champ.
std::unique_ptr<BlochPointResult> analyze_bloch_point(int current_iter, double current_time, const Matrix34d& nodes_coords, const Matrix34d& nodes_mag) {
    // Construction de la matrice de l'espace des aimantations (A_m).
    // On exprime le champ d'aimantation par rapport au premier nœud (nœud 0) choisi comme origine locale.
    Eigen::Matrix3d A_m;
    A_m.col(0) = nodes_mag.col(1) - nodes_mag.col(0); // \vec{m}_1 - \vec{m}_0
    A_m.col(1) = nodes_mag.col(2) - nodes_mag.col(0); // \vec{m}_2 - \vec{m}_0
    A_m.col(2) = nodes_mag.col(3) - nodes_mag.col(0); // \vec{m}_3 - \vec{m}_0

    // Si le déterminant est nul, les vecteurs d'aimantation sont coplanaires ou colinéaires : l'élément est dégénéré 
    // dans l'espace des phases et on ne peut pas inverser le système pour chercher un zéro unique.
    if (std::abs(A_m.determinant()) < TOL) return nullptr;
    
    // Résolution du système linéaire A_m * vec_local = -nodes_mag.col(0) pour trouver les coordonnées barycentriques
    // locales (u, v, w) où l'aimantation interpolée s'annule exactement : \vec{m}(u,v,w) = \vec{0}.
    // On utilise la factorisation QR par pivotement de colonnes (colPivHouseholderQr) pour sa robustesse numérique.
    Eigen::Vector3d vec_local = -1.0 * A_m.colPivHouseholderQr().solve(nodes_mag.col(0));

    // Vérification stricte d'inclusion : le point d'annulation \vec{m}=\vec{0} se trouve-t-il à L'INTÉRIEUR du tétraèdre courant ?
    // En coordonnées barycentriques (u, v, w), cela impose : u > 0, v > 0, w > 0 ET u + v + w < 1.
    if ((vec_local.array() > 0).all() && (vec_local.array() < 1).all() && (vec_local.sum() < 1)) {
        
        // Le point est à l'intérieur. On calcule maintenant sa position spatiale cartésienne réelle (x, y, z).
        Eigen::Matrix3d B_geo;
        B_geo.col(0) = nodes_coords.col(1) - nodes_coords.col(0); // Vecteurs colonnes géométriques \vec{r}_1 - \vec{r}_0
        B_geo.col(1) = nodes_coords.col(2) - nodes_coords.col(0); // \vec{r}_2 - \vec{r}_0
        B_geo.col(2) = nodes_coords.col(3) - nodes_coords.col(0); // \vec{r}_3 - \vec{r}_0

        // Position absolue = Position du nœud 0 + contribution linéaire des déplacements pondérés
        Eigen::Vector3d sol_cartesian = nodes_coords.col(0) + B_geo * vec_local;

        // Calcul des gradients spatiaux des composantes de l'aimantation (Jacobien de \vec{m}).
        // On pose l'équation affine complète pour chaque composante : m_i = c0 + c1*x + c2*y + c3*z.
        // On monte une matrice 4x4 combinant une colonne d'unités (pour la constante c0) et les coordonnées transposées.
        Eigen::Matrix4d A_space;
        A_space.col(0) = Eigen::Vector4d::Ones();
        A_space.block<4,3>(0,1) = nodes_coords.transpose();

        if (std::abs(A_space.determinant()) < TOL) return nullptr; // Sécurité géométrique (tétraèdre de volume nul / plat)

        // Inversion de la matrice spatiale pour extraire les coefficients polynomiaux du plan interpolateur
        Eigen::Matrix4d inv_A_space = A_space.inverse();
        Eigen::Vector4d coeff_x = inv_A_space * nodes_mag.row(0).transpose(); // Coefficients pour la composante mx
        Eigen::Vector4d coeff_y = inv_A_space * nodes_mag.row(1).transpose(); // Coefficients pour la composante my
        Eigen::Vector4d coeff_z = inv_A_space * nodes_mag.row(2).transpose(); // Coefficients pour la composante mz

        /*
        La structure des coefficients (coeff_x, coeff_y, coeff_z)
        On obtient un vecteur de 4 coefficients. Par exemple, pour mx :
        mx(x, y, z) = c0 + c1 * x + c2 * y + c3 * z
        La méthode .tail<3>() d'Eigen permet d'extraire les 3 derniers éléments d'un vecteur,
        récupérant uniquement le triplet de dérivées partielles [c1, c2, c3], c'est-à-dire le
        gradient de la composante mx : (dm_x/dx, dm_x/dy, dm_x/dz).
        */

        // Remplissage de la matrice Jacobienne J_ij = \partial m_i / \partial x_j
        Eigen::Matrix3d jac;
        jac.row(0) = coeff_x.tail<3>(); // [\partial mx / \partial x,  \partial mx / \partial y,  \partial mx / \partial z]
        jac.row(1) = coeff_y.tail<3>(); // [\partial my / \partial x,  \partial my / \partial y,  \partial my / \partial z]
        jac.row(2) = coeff_z.tail<3>(); // [\partial mz / \partial x,  \partial mz / \partial y,  \partial mz / \partial z]

        // Calcul analytique direct des composantes du vecteur Rotationnel \vec{\nabla} \times \vec{m}
        double curl_x = coeff_z[2] - coeff_y[3]; // \partial mz / \partial y - \partial my / \partial z
        double curl_y = coeff_x[3] - coeff_z[1]; // \partial mx / \partial z - \partial mz / \partial x
        double curl_z = coeff_y[1] - coeff_x[2]; // \partial my / \partial x - \partial mx / \partial y
					 
        // Analyse spectrale du Jacobien pour classifier mathématiquement la nature de la singularité.
        // Les trajectoires des lignes de champ d'aimantation autour du point dépendent des valeurs propres du Jacobien.
        Eigen::EigenSolver<Eigen::Matrix3d> es(jac);
        Eigen::Vector3cd eigvals = es.eigenvalues(); // Vecteur de 3 valeurs propres complexes (std::complex<double>)

        std::string bp_type = "unknown";
        // Vérification si toutes les valeurs propres sont purement réelles (partie imaginaire sous le seuil TOL)
        bool all_real = std::abs(eigvals[0].imag()) < TOL && std::abs(eigvals[1].imag()) < TOL && std::abs(eigvals[2].imag()) < TOL;

        if (all_real) {
            double r0 = eigvals[0].real(), r1 = eigvals[1].real(), r2 = eigvals[2].real();
            // Cas 1 : Toutes les valeurs propres sont strictement positives -> Divergence pure (Source)
            if (r0 > 0 && r1 > 0 && r2 > 0) bp_type = "source";
            // Cas 2 : Toutes les valeurs propres sont strictement négatives -> Convergence pure (Puits / Sink)
            else if (r0 < 0 && r1 < 0 && r2 < 0) bp_type = "sink";
            // Cas 3 : Signes mixtes -> Points-selles (Saddles)
            else {
                // Le produit des 3 valeurs propres donne le signe global du déterminant du Jacobien.
                // Permet de distinguer la configuration des lignes entrantes et sortantes.
                bp_type = (r0 * r1 * r2 > 0) ? "saddle 2 in - 1 out" : "saddle 1 in - 2 out";
            }
        } else {
            // Présence de valeurs propres complexes conjuguées (a \pm ib) et d'une valeur propre réelle (lamb1).
            // Traduit un comportement d'enroulement hélicoïdal ou tourbillonnaire (Spiral).
            double lamb1 = 0, a = 0;
            // Identification de la valeur propre qui est purement réelle parmi les trois
            if (std::abs(eigvals[0].imag()) < TOL) {
                lamb1 = eigvals[0].real(); a = eigvals[1].real();
            } else if (std::abs(eigvals[1].imag()) < TOL) {
                lamb1 = eigvals[1].real(); a = eigvals[0].real();
            } else {
                lamb1 = eigvals[2].real(); a = eigvals[0].real();
            }
            // Classification fine selon les signes combinés de la partie réelle complexe (a) et de la valeur réelle (lamb1)
            if (lamb1 > 0 && a > 0) bp_type = "spiral source";
            else if (lamb1 > 0 && a < 0) bp_type = "spiral saddle 2 in - 1out tail to tail";
            else if (lamb1 < 0 && a > 0) bp_type = "spiral saddle 1 in - 2out head to head";
            else if (lamb1 < 0 && a < 0) bp_type = "spiral sink";
        }

        // Renvoie l'objet conteneur de résultats sous forme de pointeur intelligent unique (std::unique_ptr)
        return std::make_unique<BlochPointResult>(BlochPointResult{
            current_iter, current_time, sol_cartesian, Eigen::Vector3d(curl_x, curl_y, curl_z), eigvals, bp_type
        });
    }
    return nullptr; // Pas de point de Bloch dans ce tétraèdre
}

// --- ANALYSE SINGULARITÉ DE SURFACE ---
// Traite les défauts topologiques bidimensionnels sur la frontière externe de l'échantillon (ex: vortex de surface, défauts de type cross/meron).
// Nécessite une projection dans le plan tangent local du triangle.
std::unique_ptr<SurfaceSingularityResult> analyze_surface_singularity(int current_iter, double current_time, const Eigen::Matrix3d& ns_coords, const Eigen::Matrix3d& ns_mag) {
    // Calcul des vecteurs tangents définissant la surface plane du triangle
    Eigen::Vector3d v1 = ns_coords.col(1) - ns_coords.col(0);
    Eigen::Vector3d v2 = ns_coords.col(2) - ns_coords.col(0);
    Eigen::Vector3d n = v1.cross(v2); // Produit vectoriel pour obtenir la normale géométrique
    double n_norm = n.norm();
    if (n_norm == 0) return nullptr; // Sécurité : évite les triangles plats d'aire nulle
    n.normalize(); // Normalisation de \vec{n} pour obtenir un vecteur unitaire

    // Création d'un repère local orthonormé direct (t0, t1, n) attaché au plan du triangle
    Eigen::Vector3d t0 = v1.normalized(); // Premier vecteur tangent unitaire aligné sur l'arête 1
    Eigen::Vector3d t1 = n.cross(t0);    // Second vecteur tangent unitaire orthogonal à t0 dans le plan

    // Projection du champ tridimensionnel d'aimantation des 3 nœuds sur ce repère local (t0, t1, n)
    // ns_mag.transpose() * t0 effectue le produit scalaire pour les 3 nœuds d'un coup.
    Eigen::Vector3d m_t0 = ns_mag.transpose() * t0; // Composantes tangentielles m_t0 au nœud 0, 1, 2
    Eigen::Vector3d m_t1 = ns_mag.transpose() * t1; // Composantes tangentielles m_t1 au nœud 0, 1, 2
    Eigen::Vector3d m_n  = ns_mag.transpose() * n;  // Composantes normales m_n au nœud 0, 1, 2

    // Projection des coordonnées 3D réelles dans le plan 2D projeté pour l'interpolation spatiale
    Eigen::Vector3d c_t0 = ns_coords.transpose() * t0;
    Eigen::Vector3d c_t1 = ns_coords.transpose() * t1;

    // Détermination de la présence d'une singularité de surface (où la projection tangentielle du champ s'annule : m_t0 = 0 et m_t1 = 0)
    Eigen::Matrix2d A_surf;
    A_surf(0, 0) = m_t0[1] - m_t0[0];
    A_surf(0, 1) = m_t1[1] - m_t1[0];
    A_surf(1, 0) = m_t0[2] - m_t0[0];
    A_surf(1, 1) = m_t1[2] - m_t1[0];

    if (std::abs(A_surf.determinant()) < TOL) return nullptr; // Système singulier non inversible

    // Résolution 2D pour trouver les facteurs de pondération (coordonnées locales dans le triangle)
    Eigen::Vector2d vec_local = -1.0 * A_surf.colPivHouseholderQr().solve(Eigen::Vector2d(m_t0[0], m_t1[0]));

    // Le point singulier de surface se trouve-t-il dans les limites physiques du triangle ?
    // Condition géométrique standard en 2D : u > 0, v > 0 ET u + v < 1.
    if (vec_local[0] > 0 && vec_local[1] > 0 && (vec_local[0] + vec_local[1] < 1)) {
        // Reconstruction tridimensionnelle de la position exacte de la singularité de surface
        Eigen::Vector3d sol_cartesian = ns_coords.col(0) + vec_local[0] * v1 + vec_local[1] * v2;
        
        // Calcul de la valeur interpolée de la composante normale de l'aimantation au point singulier
        double p_val = m_n[0] + vec_local[0] * (m_n[1] - m_n[0]) + vec_local[1] * (m_n[2] - m_n[0]);
        // Définition de la Polarité (+1 ou -1) : indique si l'aimantation sort de la surface ou y plonge au cœur du défaut
        double polarity = static_cast<double>(p_val > 0.0) - static_cast<double>(p_val < 0.0);

        // Montage du système d'interpolation linéaire 2D pour évaluer le Jacobien plan (matrice M_matrix)
        Eigen::Matrix3d M_matrix;
        M_matrix.col(0) = Eigen::Vector3d::Ones();
        M_matrix.col(1) = c_t0;
        M_matrix.col(2) = c_t1;

        if (std::abs(M_matrix.determinant()) < TOL) return nullptr;
        Eigen::Matrix3d inv_M = M_matrix.inverse();

        // Calcul des gradients bidimensionnels des composantes tangentielles de l'aimantation
        Eigen::Vector3d coeff_t0 = inv_M * m_t0; // Dérivées de m_t0 par rapport au repère plan
        Eigen::Vector3d coeff_t1 = inv_M * m_t1; // Dérivées de m_t1 par rapport au repère plan

        // Constitution du Jacobien réduit de surface (2 lignes, 2 colonnes)
        Eigen::Matrix2d jac_2d;
        jac_2d(0, 0) = coeff_t0[1]; jac_2d(0, 1) = coeff_t0[2]; // [\partial m_t0 / \partial t0,  \partial m_t0 / \partial t1]
        jac_2d(1, 0) = coeff_t1[1]; jac_2d(1, 1) = coeff_t1[2]; // [\partial m_t1 / \partial t0,  \partial m_t1 / \partial t1]

        // Résolution spectrale de la matrice 2D pour extraire les deux valeurs propres complexes
        Eigen::EigenSolver<Eigen::Matrix2d> es(jac_2d);
        Eigen::Vector2cd eigvals = es.eigenvalues();

        // Évaluation de la composante normale du rotationnel (\partial m_t1 / \partial t0 - \partial m_t0 / \partial t1)
        double curl_n = coeff_t1[1] - coeff_t0[2];
        std::string surf_type = "";

        // Une partie imaginaire non nulle traduit un enroulement circulaire/spiral (Vortex)
        bool is_complex = (std::abs(eigvals[0].imag()) > TOL || std::abs(eigvals[1].imag()) > TOL);

        if (is_complex) {
            surf_type = (eigvals[0].real() > 0) ? "spiral source" : "spiral sink";
        } else {
            // Valeurs propres réelles : si elles sont de signes opposés, c'est un col (Saddle / Anti-vortex de surface)
            if ((eigvals[0].real() > 0 && eigvals[1].real() < 0) || (eigvals[0].real() < 0 && eigvals[1].real() > 0)) {
                surf_type = "saddle";
            } else {
                // Sinon, c'est un nœud divergent (Source) ou convergent (Sink)
                // les deux valeurs propres sont réelles ET de le même signe
                surf_type = (eigvals[0].real() > 0) ? "source" : "sink";
            }
        }

        return std::make_unique<SurfaceSingularityResult>(SurfaceSingularityResult{
            current_iter, current_time, sol_cartesian, curl_n, eigvals, surf_type, polarity
        });
    }
    return nullptr; // Pas de singularité détectée sur cette facette de surface
}

// --- MAIN ---
int main(int argc, char* argv[]) {
    // 0. Récupération du fichier maillage (.msh) depuis la ligne de commande
    if (argc < 2) {
        std::cerr << "Erreur : Nom du fichier maillage (.msh) manquant.\n";
        std::cerr << "Usage   : " << argv[0] << " <chemin_du_fichier_maillage.msh>\n";
        return 1;
    }
    std::string msh_file = argv[1];

    // 1. Lister tous les fichiers sol et extraire l'indice d'itération via Expressions Régulières (Regex)
    std::vector<std::pair<int, std::string>> sol_files;
    
    std::cout << "Recherche des fichiers solutions dans '" << DIRECTORY_PATH << "'...\n";
    if (!fs::exists(DIRECTORY_PATH)) {
        std::cerr << "Dossier '" << DIRECTORY_PATH << "' inexistant.\n";
        return 1;
    }

    // Parcours du système de fichiers via la bibliothèque standard <filesystem> de C++17
    for (const auto& entry : fs::directory_iterator(DIRECTORY_PATH)) {
        std::string filename = entry.path().filename().string();
        
        // Filtrage strict sur l'extension .sol
        if (entry.path().extension() == ".sol") {
            try {
                // Regex configurée pour intercepter le motif de nommage, ex: "magn_iter450.sol"
                // (\d+) capture un ou plusieurs chiffres consécutifs formant l'index d'itération
                std::regex re(R"(.*_iter(\d+)\.sol)");
                std::smatch match;
                int num = -1;
                if (std::regex_match(filename, match, re)) {
                    num = std::stoi(match[1].str()); // Conversion de la capture texte en entier pur
                }
                sol_files.push_back({num, entry.path().string()});
            } catch (...) {
                // Gestion de secours si le format du nom dévie mais possède la bonne extension
                sol_files.push_back({0, entry.path().string()});
            }
        }
    }

    if (sol_files.empty()) {
        std::cerr << "Aucun fichier solution (*.sol) trouvé.\n";
        return 1;
    }

    // Tri des fichiers par ordre chronologique (grâce au 'std::pair', le tri s'effectue naturellement sur le premier membre : l'entier 'iter')
    std::sort(sol_files.begin(), sol_files.end());
    std::cout << sol_files.size() << " fichiers solutions trouvés et triés.\n";

    // 2. Pré-chargement de la première solution pour connaître la taille attendue du maillage.
    // Permet une allocation mémoire optimisée et une barrière de contrôle pour valider la cohérence avec le fichier Gmsh.
    Eigen::MatrixXd initial_mag;
    double dummy_time = 0.0;
    try {
        initial_mag = load_magnetization(sol_files[0].second, dummy_time);
    } catch (const std::exception& e) {
        std::cerr << "Erreur lors du pré-chargement : " << e.what() << "\n";
        return 1;
    }

    // 3. CHARGEMENT ET CORRECTION DU MAILLAGE (UNE SEULE FOIS)
    // Mesure du temps CPU hautement précise via la bibliothèque <chrono>
    auto mesh_start = std::chrono::high_resolution_clock::now();
    std::cout << "Chargement unique du maillage : " << msh_file << "...\n";
    Mesh mesh;
    try {
        // initial_mag.rows() fournit le nombre attendu de nœuds pour dimensionner la matrice de points
        mesh = load_mesh_gmsh(msh_file, initial_mag.rows());
    } catch (const std::exception& e) {
        std::cerr << "Erreur de maillage : " << e.what() << "\n";
        return 1;
    }
    // Appel obligatoire de la fonction d'orientation pour corriger les normales de surface face au volume
    mesh.triangles = orient_triangles_outward(mesh.points, mesh.tetrahedrons, mesh.triangles);
    auto mesh_end = std::chrono::high_resolution_clock::now(); // <-- FIN CHRONO
    std::chrono::duration<double> mesh_duration = mesh_end - mesh_start;
    std::cout << "=> Temps de traitement du maillage : " << mesh_duration.count() << " secondes.\n";

    // Conteneurs globaux pour accumuler tous les pas de temps avant écriture finale sur disque
    std::vector<BlochPointResult> global_bloch_points;
    std::vector<SurfaceSingularityResult> global_surface_singularities;

    // 4. BOUCLE PRINCIPALE SUR TOUTES LES SOLUTIONS CHRONOLOGIQUES
    auto sol_start = std::chrono::high_resolution_clock::now(); 
    for (const auto& [iter, file_path] : sol_files) {
        double current_time = 0.0;
        Eigen::MatrixXd mag;
        try {
            mag = load_magnetization(file_path, current_time);
        } catch (const std::exception& e) {
            std::cerr << "Échec de lecture pour " << file_path << " (" << e.what() << "), ignoré.\n";
            continue;
        }

        std::cout << "\n---------------------------------------------\n";
        std::cout << "Traitement de : " << file_path << " (Iteration: " << iter << " | Time: " << current_time << ")...\n";

        // Garde-fou numérique : vérifie si le fichier d'aimantation lu est compatible avec la géométrie du maillage chargé
        if (mag.rows() != mesh.points.rows()) {
            std::cerr << "Incohérence de taille de nœuds dans " << file_path << ", ignoré.\n";
            continue;
        }

        // --- ANALYSE VOLUME (Parcours exhaustif de tous les tétraèdres du domaine) ---
        for (const auto& nodes_idx : mesh.tetrahedrons) {
            Matrix34d t_coords, t_mag; 
            // Extraction locale des coordonnées et aimantations des 4 sommets du tétraèdre courant
            for(int i = 0; i < 4; ++i) {
                t_coords.col(i) = mesh.points.row(nodes_idx[i]);
                t_mag.col(i) = mag.row(nodes_idx[i]);
            }

            // Test de pré-filtrage topologique ultra-rapide (sign_check) utilisant les expressions Lambdas de C++.
            // Un point de Bloch (\vec{m}=\vec{0}) ne peut exister dans un tétraèdre QUE si chaque composante (mx, my, mz) 
            // change de signe au moins une fois parmi les 4 nœuds de l'élément. 
            // Si une composante reste entièrement positive ou entièrement négative, l'aimantation ne peut pas s'annuler à l'intérieur.
            auto sign_check = [](const Matrix34d& m, int axis) {
                bool has_pos = false, has_neg = false;
                for(int i = 0; i < 4; ++i) {
                    if(m(axis, i) > 0) has_pos = true;
                    if(m(axis, i) < 0) has_neg = true;
                }
                return has_pos && has_neg; // Vrai si croisement du zéro
            };

            // On n'exécute la lourde machinerie d'inversion matricielle d'analyze_bloch_point que si le pré-test est validé pour X, Y, et Z.
            if (sign_check(t_mag, 0) && sign_check(t_mag, 1) && sign_check(t_mag, 2)) {
                auto res_bp = analyze_bloch_point(iter, current_time, t_coords, t_mag);
                if (res_bp) {
                    std::printf(" -> [BP Volume] détecté à : [%.5f, %.5f, %.5f] nm (%s)\n", 
                                res_bp->pos.x(), res_bp->pos.y(), res_bp->pos.z(), res_bp->type.c_str());
                    global_bloch_points.push_back(*res_bp); // Déréférencement du pointeur pour stockage par copie dans le tableau global
                }
            }
        }

        // --- ANALYSE SURFACE (Parcours de toutes les facettes triangulaires frontières) ---
        for (const auto& nodes_idx : mesh.triangles) {
            Eigen::Matrix3d s_coords, s_mag; 
            // Extraction des données nodales locales pour les 3 sommets du triangle courant
            for(int i = 0; i < 3; ++i) {
                s_coords.col(i) = mesh.points.row(nodes_idx[i]);
                s_mag.col(i) = mag.row(nodes_idx[i]);
            }

            auto res_surf = analyze_surface_singularity(iter, current_time, s_coords, s_mag);
            if (res_surf) {
                std::printf(" -> [Singularité Surface] (%s) à : [%.5f, %.5f, %.5f] nm | Polarité: %.1f\n",
                            res_surf->type.c_str(), res_surf->pos.x(), res_surf->pos.y(), res_surf->pos.z(), res_surf->polarity);
                global_surface_singularities.push_back(*res_surf);
            }
        }
    }

    auto sol_end = std::chrono::high_resolution_clock::now(); 
    std::chrono::duration<double> sol_duration = sol_end - sol_start;
    std::cout << "\n=> Temps total de traitement de tous les fichiers .sol : "
              << sol_duration.count() << " secondes.\n";

    // 5. SAUVEGARDE GLOBALE COMPLETE (Génération des fichiers de rapports tabulés structurés)
    std::cout << "\n---------------------------------------------\nExécution de l'export final...\n";
    
    // Exportation du fichier texte compilant tous les points de Bloch (Volume) trouvés au cours du temps
    std::ofstream f_vol("all_volume_bloch_points.txt");
    if (f_vol.is_open()) {
        // Écriture d'une entête alignée de manière rigoureuse avec les manipulateurs de flux (std::setw)
        f_vol << std::left  << std::setw(8)  << "#iter"
              << std::setw(15) << "time"
              << std::right << std::setw(10) << "x" << std::setw(10) << "y" << std::setw(10) << "z"
              << std::setw(10) << "curl_x" << std::setw(10) << "curl_y" << std::setw(10) << "curl_z"
              << std::setw(12) << "eig0_re" << std::setw(12) << "eig0_im"
              << std::setw(12) << "eig1_re" << std::setw(12) << "eig1_im"
              << std::setw(12) << "eig2_re" << std::setw(12) << "eig2_im"
              << "  type\n";

        for (const auto& bp : global_bloch_points) {
            f_vol << std::left  << std::setw(8)  << bp.iter
                  << std::scientific << std::setprecision(6) << std::setw(15) << bp.time // Temps au format scientifique
                  << std::right << std::fixed << std::setprecision(4)                     // Coordonnées et scalaires au format fixe
                  << std::setw(10) << bp.pos.x()  << std::setw(10) << bp.pos.y()  << std::setw(10) << bp.pos.z()
                  << std::setw(10) << bp.curl.x() << std::setw(10) << bp.curl.y() << std::setw(10) << bp.curl.z()
                  << std::setw(12) << bp.eigvals[0].real() << std::setw(12) << bp.eigvals[0].imag()
                  << std::setw(12) << bp.eigvals[1].real() << std::setw(12) << bp.eigvals[1].imag()
                  << std::setw(12) << bp.eigvals[2].real() << std::setw(12) << bp.eigvals[2].imag()
                  << "  " << std::left << bp.type << '\n';
        }
    }

    // Exportation du fichier texte compilant toutes les singularités de surface trouvées au cours du temps
    std::ofstream f_surf("all_surface_singularities.txt");
    if (f_surf.is_open()) {
        f_surf << std::left  << std::setw(8)  << "#iter"
               << std::setw(15) << "time"
               << std::right << std::setw(10) << "x" << std::setw(10) << "y" << std::setw(10) << "z"
               << std::setw(12) << "curl_n" 
               << std::setw(10) << "polarity"
               << std::setw(12) << "eig0_re" << std::setw(12) << "eig0_im"
               << std::setw(12) << "eig1_re" << std::setw(12) << "eig1_im"
               << "  type\n";

        for(const auto& ss : global_surface_singularities) {
            f_surf << std::left  << std::setw(8)  << ss.iter
                   << std::scientific << std::setprecision(6) << std::setw(15) << ss.time
                   << std::right << std::fixed << std::setprecision(4)
                   << std::setw(10) << ss.pos.x() << std::setw(10) << ss.pos.y() << std::setw(10) << ss.pos.z()
                   << std::setw(12) << ss.curl_n
                   << std::setw(10) << std::setprecision(1) << ss.polarity
                   << std::fixed << std::setprecision(4)
                   << std::setw(12) << ss.eigvals[0].real() << std::setw(12) << ss.eigvals[0].imag()
                   << std::setw(12) << ss.eigvals[1].real() << std::setw(12) << ss.eigvals[1].imag()
                   << "  " << std::left << ss.type << '\n';
        }
    }

    // Affichage final des performances et bilan d'exécution sur la console
    std::cout << "Analyses multi-fichiers et exports complets terminés.\n";
    std::cout << "\n===== BILAN DES TEMPS DE CALCUL =====" << std::endl;
    std::cout << "Lecture & correction maillage : " << mesh_duration.count() << " s" << std::endl;
    std::cout << "Traitement des fichiers .sol   : " << sol_duration.count() << " s" << std::endl;
    std::cout << "Temps calcul total d'analyse  : " << (mesh_duration.count() + sol_duration.count()) << " s" << std::endl;
    std::cout << "=====================================\n" << std::endl;
    return 0; // Fin normale du programme sans erreur
}
