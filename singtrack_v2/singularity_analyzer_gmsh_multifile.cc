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

// Inclusion de l'API Gmsh
#include <gmsh.h>

namespace fs = std::filesystem;
using Matrix34d = Eigen::Matrix<double, 3, 4>;

// --- CONFIGURATION & PARAMÈTRES ---
const std::string DIRECTORY_PATH = "."; // Dossier où se trouvent les fichiers *.sol ('.' signifie dossier courant)
const double TOL = 1e-16;

struct Mesh {
    Eigen::MatrixXd points;                  // (N, 3)
    std::vector<Eigen::Vector4i> tetrahedrons; // Liste des indices (4 nœuds)
    std::vector<Eigen::Vector3i> triangles;    // Liste des indices (3 nœuds)
};

// Structures de résultats complètes
struct BlochPointResult {
    int iter;
    double time;
    Eigen::Vector3d pos;
    Eigen::Vector3d curl;
    Eigen::Vector3cd eigvals;
    std::string type;
};

struct SurfaceSingularityResult {
    int iter;
    double time;
    Eigen::Vector3d pos;
    double curl_n;
    Eigen::Vector2cd eigvals;
    std::string type;
    double polarity;
};

// --- FONCTION 1 : LECTURE D'UN FICHIER SOLUTION ---
Eigen::MatrixXd load_magnetization(const std::string& sol_filename, double& out_time) {
    std::ifstream sol_file(sol_filename);
    if (!sol_file.is_open()) throw std::runtime_error("Fichier SOL manquant : " + sol_filename);
    
    std::string line;
    // Lire la première ligne pour extraire le temps (ex: #time :   +7.3222568275e-04)
    if (!std::getline(sol_file, line)) {
        throw std::runtime_error("Fichier SOL vide ou invalide : " + sol_filename);
    }

    // Extraction de la valeur numérique du temps après "#time :"
    size_t pos_colon = line.find(':');
    if (pos_colon != std::string::npos) {
        std::string time_str = line.substr(pos_colon + 1);
        try {
            out_time = std::stod(time_str);
        } catch (...) {
            out_time = 0.0;
            std::cerr << "Attention : Impossible de parser la valeur de temps dans " << sol_filename << ". Fixé à 0.0.\n";
        }
    } else {
        out_time = 0.0;
        std::cerr << "Attention : En-tête de temps mal formé dans " << sol_filename << ". Fixé à 0.0.\n";
    }

    std::vector<Eigen::Vector3d> mag_list;
    
    // Lire ligne par ligne à partir de la deuxième ligne
    while (std::getline(sol_file, line)) {
        if (line.empty() || line[0] == '#') continue; // Ignorer lignes vides ou commentaires subsidiaires
        
        std::stringstream ss(line);
        std::string val;
        std::vector<double> columns;
        
        // Extraction de toutes les colonnes de la ligne courante
        while (ss >> val) {
            try {
                columns.push_back(std::stod(val));
            } catch (...) {
                throw std::runtime_error("Erreur de format numérique dans le fichier : " + sol_filename);
            }
        }
        
        // Vérification qu'on a au moins les 7 colonnes nécessaires
        if (columns.size() < 7) {
            throw std::runtime_error("Ligne incomplète dans " + sol_filename + " (Moins de 7 colonnes trouvées).");
        }
        
        // Récupération des colonnes 5, 6 et 7 (indices 4, 5, 6)
        double mx = columns[4];
        double my = columns[5];
        double mz = columns[6];
        
        mag_list.push_back(Eigen::Vector3d(mx, my, mz));
    }
    sol_file.close();

    Eigen::MatrixXd mag(mag_list.size(), 3);
    for (size_t i = 0; i < mag_list.size(); ++i) {
        mag.row(i) = mag_list[i];
    }
    return mag;
}

// --- FONCTION 2 : LECTURE DU MAILLAGE VIA GMSH API (AVEC SQUIZZAGE DES LIGNES HORS-FORMAT) ---
Mesh load_mesh_gmsh(const std::string& msh_filename, size_t expected_nodes) {
    Mesh mesh;

    // 1. Pré-traitement et nettoyage du fichier maillage
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
        // Détection du tag de fin du format de maillage
        if (line.find("$EndMeshFormat") != std::string::npos) {
            dst_file << line << "\n";
            inside_corrupted_zone = true; // On commence à surveiller le saut de lignes
            continue;
        }

        // Détection du tag réglementaire suivant ($Nodes)
        if (line.find("$Nodes") != std::string::npos) {
            inside_corrupted_zone = false; // Fin de la zone de lignes parasites
        }

        // Si on est entre les deux et que la ligne n'est pas le début de $Nodes, on la squizze
        if (inside_corrupted_zone) {
            skipped_lines_count++;
            continue; // Ignore la ligne
        }

        // Écrit les lignes valides dans le fichier temporaire
        dst_file << line << "\n";
    }
    src_file.close();
    dst_file.close();

    // Avertir l'utilisateur si des lignes ont été ignorées
    if (skipped_lines_count > 0) {
        std::cout << "\n[ATTENTION] Le fichier maillage " << msh_filename << " contenait " 
                  << skipped_lines_count << " ligne(s) non standard entre $EndMeshFormat et $Nodes.\n"
                  << "            Ces lignes ont ete squizees pour assurer une lecture correcte.\n\n";
    }

    // 2. Initialisation de Gmsh et ouverture du fichier nettoyé
    gmsh::initialize();
    gmsh::option::setNumber("General.Terminal", 0);
    
    try {
        gmsh::open(clean_filename);
    } catch (...) {
        std::filesystem::remove(clean_filename); // Nettoyage
        gmsh::finalize();
        throw std::runtime_error("Impossible d'ouvrir le fichier maillage via Gmsh : " + msh_filename);
    }

    // Suppression du fichier temporaire une fois chargé en mémoire par Gmsh
    std::filesystem::remove(clean_filename);

    // 3. Extraction classique des nœuds et éléments
    std::vector<size_t> nodeTags;
    std::vector<double> coord;
    std::vector<double> parametricCoord;
    gmsh::model::mesh::getNodes(nodeTags, coord, parametricCoord);

    if (nodeTags.empty()) {
        gmsh::finalize();
        throw std::runtime_error("Le maillage chargé ne contient aucun nœud.");
    }

    mesh.points.resize(expected_nodes, 3);

    for (size_t i = 0; i < nodeTags.size(); ++i) {
        size_t tag = nodeTags[i];
        size_t idx = tag - 1; 

        if (idx < expected_nodes) {
            mesh.points(idx, 0) = coord[3 * i];
            mesh.points(idx, 1) = coord[3 * i + 1];
            mesh.points(idx, 2) = coord[3 * i + 2];
        }
    }

    std::vector<int> elementTypes;
    std::vector<std::vector<size_t>> elementTags;
    std::vector<std::vector<size_t>> nodeTagsPerElement;
    gmsh::model::mesh::getElements(elementTypes, elementTags, nodeTagsPerElement);

    for (size_t i = 0; i < elementTypes.size(); ++i) {
        int type = elementTypes[i];
        const auto& nodes = nodeTagsPerElement[i];

        if (type == 2) { // Triangle
            for (size_t j = 0; j < nodes.size(); j += 3) {
                int idx0 = static_cast<int>(nodes[j] - 1);
                int idx1 = static_cast<int>(nodes[j+1] - 1);
                int idx2 = static_cast<int>(nodes[j+2] - 1);
                mesh.triangles.push_back(Eigen::Vector3i(idx0, idx1, idx2));
            }
        } 
        else if (type == 4) { // Tétraèdre
            for (size_t j = 0; j < nodes.size(); j += 4) {
                int idx0 = static_cast<int>(nodes[j] - 1);
                int idx1 = static_cast<int>(nodes[j+1] - 1);
                int idx2 = static_cast<int>(nodes[j+2] - 1);
                int idx3 = static_cast<int>(nodes[j+3] - 1);
                mesh.tetrahedrons.push_back(Eigen::Vector4i(idx0, idx1, idx2, idx3));
            }
        }
    }

    gmsh::finalize();

    if (mesh.tetrahedrons.empty()) throw std::runtime_error("Aucun élément tétraédrique trouvé.");
    return mesh;
}

// --- RÉORIENTATION DES TRIANGLES ---
std::vector<Eigen::Vector3i> orient_triangles_outward(const Eigen::MatrixXd& points, 
                                                      const std::vector<Eigen::Vector4i>& tetrahedrons, 
                                                      const std::vector<Eigen::Vector3i>& triangles) {
    if (triangles.empty()) return triangles;
    std::cout << "Structure de recherche des tétraèdres en cours... " << std::endl;
    std::map<int, std::vector<int>> node_to_tetra;
    for (int t_idx = 0; t_idx < static_cast<int>(tetrahedrons.size()); ++t_idx) {
        for (int i = 0; i < 4; ++i) {
            node_to_tetra[tetrahedrons[t_idx][i]].push_back(t_idx);
        }
    }
    std::cout << "Correction de l'orientation des triangles..." << std::endl;
    std::vector<Eigen::Vector3i> oriented_triangles = triangles;
    int corrected_count = 0;
    for (size_t i = 0; i < oriented_triangles.size(); ++i) {
        auto& tri = oriented_triangles[i];
        const auto& t0_list = node_to_tetra[tri[0]];
        const auto& t1_list = node_to_tetra[tri[1]];
        const auto& t2_list = node_to_tetra[tri[2]];
        std::set<int> intersect1, final_intersect;
        std::set_intersection(t0_list.begin(), t0_list.end(), t1_list.begin(), t1_list.end(), std::inserter(intersect1, intersect1.begin()));
        std::set_intersection(intersect1.begin(), intersect1.end(), t2_list.begin(), t2_list.end(), std::inserter(final_intersect, final_intersect.begin()));
        if (final_intersect.empty()) continue;
        int tetra_idx = *final_intersect.begin();
        Eigen::Vector4i tetra = tetrahedrons[tetra_idx];
        int internal_node = -1;
        for (int n = 0; n < 4; ++n) {
            if (tetra[n] != tri[0] && tetra[n] != tri[1] && tetra[n] != tri[2]) {
                internal_node = tetra[n];
                break;
            }
        }
        Eigen::Vector3d p0 = points.row(tri[0]);
        Eigen::Vector3d p1 = points.row(tri[1]);
        Eigen::Vector3d p2 = points.row(tri[2]);
        Eigen::Vector3d p_in = points.row(internal_node);
        Eigen::Vector3d v1 = p1 - p0;
        Eigen::Vector3d v2 = p2 - p0;
        Eigen::Vector3d normal = v1.cross(v2);
        Eigen::Vector3d v_to_internal = p_in - p0;
        if (normal.dot(v_to_internal) > 0) {
            std::swap(tri[0], tri[1]);
            corrected_count++;
        }
    }
    std::cout << "Orientation terminée. " << corrected_count << " / " << triangles.size() << " triangles inversés.\n";
    return oriented_triangles;
}

// --- ANALYSE POINT DE BLOCH (VOLUME) ---
std::unique_ptr<BlochPointResult> analyze_bloch_point(int current_iter, double current_time, const Matrix34d& nodes_coords, const Matrix34d& nodes_mag) {
    Eigen::Matrix3d A_m;
    A_m.col(0) = nodes_mag.col(1) - nodes_mag.col(0);
    A_m.col(1) = nodes_mag.col(2) - nodes_mag.col(0);
    A_m.col(2) = nodes_mag.col(3) - nodes_mag.col(0);

    if (std::abs(A_m.determinant()) < TOL) return nullptr;
    Eigen::Vector3d vec_local = -1.0 * A_m.colPivHouseholderQr().solve(nodes_mag.col(0));

    if ((vec_local.array() > 0).all() && (vec_local.array() < 1).all() && (vec_local.sum() < 1)) {
        Eigen::Matrix3d B_geo;
        B_geo.col(0) = nodes_coords.col(1) - nodes_coords.col(0);
        B_geo.col(1) = nodes_coords.col(2) - nodes_coords.col(0);
        B_geo.col(2) = nodes_coords.col(3) - nodes_coords.col(0);

        Eigen::Vector3d sol_cartesian = nodes_coords.col(0) + B_geo * vec_local;

        Eigen::Matrix4d A_space;
        A_space.col(0) = Eigen::Vector4d::Ones();
        A_space.block<4,3>(0,1) = nodes_coords.transpose();

        if (std::abs(A_space.determinant()) < TOL) return nullptr;

        Eigen::Matrix4d inv_A_space = A_space.inverse();
        Eigen::Vector4d coeff_x = inv_A_space * nodes_mag.row(0).transpose();
        Eigen::Vector4d coeff_y = inv_A_space * nodes_mag.row(1).transpose();
        Eigen::Vector4d coeff_z = inv_A_space * nodes_mag.row(2).transpose();

        Eigen::Matrix3d jac;
        jac.row(0) = coeff_x.tail<3>();
        jac.row(1) = coeff_y.tail<3>();
        jac.row(2) = coeff_z.tail<3>();

        double curl_x = coeff_y[3] - coeff_z[2];
        double curl_y = coeff_z[1] - coeff_x[3];
        double curl_z = coeff_x[2] - coeff_y[1];

        Eigen::EigenSolver<Eigen::Matrix3d> es(jac);
        Eigen::Vector3cd eigvals = es.eigenvalues();

        std::string bp_type = "unknown";
        bool all_real = std::abs(eigvals[0].imag()) < TOL && std::abs(eigvals[1].imag()) < TOL && std::abs(eigvals[2].imag()) < TOL;

        if (all_real) {
            double r0 = eigvals[0].real(), r1 = eigvals[1].real(), r2 = eigvals[2].real();
            if (r0 > 0 && r1 > 0 && r2 > 0) bp_type = "source";
            else if (r0 < 0 && r1 < 0 && r2 < 0) bp_type = "sink";
            else {
                bp_type = (r0 * r1 * r2 > 0) ? "saddle 2 in - 1 out" : "saddle 1 in - 2 out";
            }
        } else {
            double lamb1 = 0, a = 0;
            if (std::abs(eigvals[0].imag()) < TOL) {
                lamb1 = eigvals[0].real(); a = eigvals[1].real();
            } else if (std::abs(eigvals[1].imag()) < TOL) {
                lamb1 = eigvals[1].real(); a = eigvals[0].real();
            } else {
                lamb1 = eigvals[2].real(); a = eigvals[0].real();
            }
            if (lamb1 > 0 && a > 0) bp_type = "spiral source";
            else if (lamb1 > 0 && a < 0) bp_type = "spiral saddle 2 in - 1out tail to tail";
            else if (lamb1 < 0 && a > 0) bp_type = "spiral saddle 1 in - 2out head to head";
            else if (lamb1 < 0 && a < 0) bp_type = "spiral sink";
        }

        return std::make_unique<BlochPointResult>(BlochPointResult{
            current_iter, current_time, sol_cartesian, Eigen::Vector3d(curl_x, curl_y, curl_z), eigvals, bp_type
        });
    }
    return nullptr;
}

// --- ANALYSE SINGULARITÉ DE SURFACE ---
std::unique_ptr<SurfaceSingularityResult> analyze_surface_singularity(int current_iter, double current_time, const Eigen::Matrix3d& ns_coords, const Eigen::Matrix3d& ns_mag) {
    Eigen::Vector3d v1 = ns_coords.col(1) - ns_coords.col(0);
    Eigen::Vector3d v2 = ns_coords.col(2) - ns_coords.col(0);
    Eigen::Vector3d n = v1.cross(v2);
    double n_norm = n.norm();
    if (n_norm == 0) return nullptr;
    n.normalize();

    Eigen::Vector3d t0 = v1.normalized();
    Eigen::Vector3d t1 = n.cross(t0);

    Eigen::Vector3d m_t0 = ns_mag.transpose() * t0;
    Eigen::Vector3d m_t1 = ns_mag.transpose() * t1;
    Eigen::Vector3d m_n  = ns_mag.transpose() * n;

    Eigen::Vector3d c_t0 = ns_coords.transpose() * t0;
    Eigen::Vector3d c_t1 = ns_coords.transpose() * t1;

    Eigen::Matrix2d A_surf;
    A_surf(0, 0) = m_t0[1] - m_t0[0];
    A_surf(0, 1) = m_t1[1] - m_t1[0];
    A_surf(1, 0) = m_t0[2] - m_t0[0];
    A_surf(1, 1) = m_t1[2] - m_t1[0];

    if (std::abs(A_surf.determinant()) < TOL) return nullptr;

    Eigen::Vector2d vec_local = -1.0 * A_surf.colPivHouseholderQr().solve(Eigen::Vector2d(m_t0[0], m_t1[0]));

    if (vec_local[0] > 0 && vec_local[1] > 0 && (vec_local[0] + vec_local[1] < 1)) {
        Eigen::Vector3d sol_cartesian = ns_coords.col(0) + vec_local[0] * v1 + vec_local[1] * v2;
        
        double p_val = m_n[0] + vec_local[0] * (m_n[1] - m_n[0]) + vec_local[1] * (m_n[2] - m_n[0]);
        double polarity = static_cast<double>(p_val > 0.0) - static_cast<double>(p_val < 0.0);

        Eigen::Matrix3d M_matrix;
        M_matrix.col(0) = Eigen::Vector3d::Ones();
        M_matrix.col(1) = c_t0;
        M_matrix.col(2) = c_t1;

        if (std::abs(M_matrix.determinant()) < TOL) return nullptr;
        Eigen::Matrix3d inv_M = M_matrix.inverse();

        Eigen::Vector3d coeff_t0 = inv_M * m_t0;
        Eigen::Vector3d coeff_t1 = inv_M * m_t1;

        Eigen::Matrix2d jac_2d;
        jac_2d(0, 0) = coeff_t0[1]; jac_2d(0, 1) = coeff_t0[2];
        jac_2d(1, 0) = coeff_t1[1]; jac_2d(1, 1) = coeff_t1[2];

        Eigen::EigenSolver<Eigen::Matrix2d> es(jac_2d);
        Eigen::Vector2cd eigvals = es.eigenvalues();

        double curl_n = coeff_t1[1] - coeff_t0[2];
        std::string surf_type = "";

        bool is_complex = (std::abs(eigvals[0].imag()) > TOL || std::abs(eigvals[1].imag()) > TOL);

        if (is_complex) {
            surf_type = (eigvals[0].real() > 0) ? "spiral source" : "spiral sink";
        } else {
            if ((eigvals[0].real() > 0 && eigvals[1].real() < 0) || (eigvals[0].real() < 0 && eigvals[1].real() > 0)) {
                surf_type = "saddle";
            } else {
                surf_type = (eigvals[0].real() > 0) ? "source" : "sink";
            }
        }

        return std::make_unique<SurfaceSingularityResult>(SurfaceSingularityResult{
            current_iter, current_time, sol_cartesian, curl_n, eigvals, surf_type, polarity
        });
    }
    return nullptr;
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

    // 1. Lister tous les fichiers sol*.in et extraire l'indice d'itération
    std::vector<std::pair<int, std::string>> sol_files;
    
    std::cout << "Recherche des fichiers solutions dans '" << DIRECTORY_PATH << "'...\n";
    if (!fs::exists(DIRECTORY_PATH)) {
        std::cerr << "Dossier '" << DIRECTORY_PATH << "' inexistant.\n";
        return 1;
    }

    for (const auto& entry : fs::directory_iterator(DIRECTORY_PATH)) {
        std::string filename = entry.path().filename().string();
        
        if (entry.path().extension() == ".sol") {
            try {
                std::regex re(R"(.*_iter(\d+)\.sol)");
                std::smatch match;
                int num = -1;
                if (std::regex_match(filename, match, re)) {
                    num = std::stoi(match[1].str());
                }
                sol_files.push_back({num, entry.path().string()});
            } catch (...) {
                sol_files.push_back({0, entry.path().string()});
            }
        }
    }

    if (sol_files.empty()) {
        std::cerr << "Aucun fichier solution (*.sol) trouvé.\n";
        return 1;
    }

    // Tri des fichiers par ordre chronologique
    std::sort(sol_files.begin(), sol_files.end());
    std::cout << sol_files.size() << " fichiers solutions trouvés et triés.\n";

    // 2. Pré-chargement de la première solution pour connaître la taille attendue du maillage
    Eigen::MatrixXd initial_mag;
    double dummy_time = 0.0;
    try {
        initial_mag = load_magnetization(sol_files[0].second, dummy_time);
    } catch (const std::exception& e) {
        std::cerr << "Erreur lors du pré-chargement : " << e.what() << "\n";
        return 1;
    }

    // 3. CHARGEMENT ET CORRECTION DU MAILLAGE (UNE SEULE FOIS)
    std::cout << "Chargement unique du maillage : " << msh_file << "...\n";
    Mesh mesh;
    try {
        mesh = load_mesh_gmsh(msh_file, initial_mag.rows());
    } catch (const std::exception& e) {
        std::cerr << "Erreur de maillage : " << e.what() << "\n";
        return 1;
    }
    mesh.triangles = orient_triangles_outward(mesh.points, mesh.tetrahedrons, mesh.triangles);

    // Conteneurs globaux pour accumuler tous les pas de temps
    std::vector<BlochPointResult> global_bloch_points;
    std::vector<SurfaceSingularityResult> global_surface_singularities;

    // 4. BOUCLE SUR TOUTES LES SOLUTIONS
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

        if (mag.rows() != mesh.points.rows()) {
            std::cerr << "Incohérence de taille de nœuds dans " << file_path << ", ignoré.\n";
            continue;
        }

        // --- ANALYSE VOLUME ---
        for (const auto& nodes_idx : mesh.tetrahedrons) {
            Matrix34d t_coords, t_mag; 
            for(int i = 0; i < 4; ++i) {
                t_coords.col(i) = mesh.points.row(nodes_idx[i]);
                t_mag.col(i) = mag.row(nodes_idx[i]);
            }

            auto sign_check = [](const Matrix34d& m, int axis) {
                bool has_pos = false, has_neg = false;
                for(int i = 0; i < 4; ++i) {
                    if(m(axis, i) > 0) has_pos = true;
                    if(m(axis, i) < 0) has_neg = true;
                }
                return has_pos && has_neg;
            };

            if (sign_check(t_mag, 0) && sign_check(t_mag, 1) && sign_check(t_mag, 2)) {
                auto res_bp = analyze_bloch_point(iter, current_time, t_coords, t_mag);
                if (res_bp) {
                    std::printf(" -> [BP Volume] détecté à : [%.5f, %.5f, %.5f] nm (%s)\n", 
                                res_bp->pos.x(), res_bp->pos.y(), res_bp->pos.z(), res_bp->type.c_str());
                    global_bloch_points.push_back(*res_bp);
                }
            }
        }

        // --- ANALYSE SURFACE ---
        for (const auto& nodes_idx : mesh.triangles) {
            Eigen::Matrix3d s_coords, s_mag; 
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

    // 5. SAUVEGARDE GLOBALE COMPLETE
    std::cout << "\n---------------------------------------------\nExécution de l'export final...\n";
    
    std::ofstream f_vol("all_volume_bloch_points.txt");
    if (f_vol.is_open()) {
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
                  << std::scientific << std::setprecision(6) << std::setw(15) << bp.time
                  << std::right << std::fixed << std::setprecision(4)
                  << std::setw(10) << bp.pos.x()  << std::setw(10) << bp.pos.y()  << std::setw(10) << bp.pos.z()
                  << std::setw(10) << bp.curl.x() << std::setw(10) << bp.curl.y() << std::setw(10) << bp.curl.z()
                  << std::setw(12) << bp.eigvals[0].real() << std::setw(12) << bp.eigvals[0].imag()
                  << std::setw(12) << bp.eigvals[1].real() << std::setw(12) << bp.eigvals[1].imag()
                  << std::setw(12) << bp.eigvals[2].real() << std::setw(12) << bp.eigvals[2].imag()
                  << "  " << std::left << bp.type << '\n';
        }
    }

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

    std::cout << "Analyses multi-fichiers et exports complets terminés.\n";
    return 0;
}
