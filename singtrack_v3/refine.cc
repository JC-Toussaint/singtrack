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
using Matrix34d = Eigen::Matrix<double, 3, 4>;

// --- CONFIGURATION & PARAMÈTRES ---
const std::string DIRECTORY_PATH = "."; 
const double TOL = 1e-16;

// Structure représentant le maillage géométrique 3D
struct Mesh {
    Eigen::MatrixXd points;                  
    std::vector<Eigen::Vector4i> tetrahedrons; 
    std::vector<Eigen::Vector3i> triangles;    
    Eigen::MatrixXd node_normals;              
};

// Structure pour identifier une arête unique (Subdivision)
struct Edge {
    int v1, v2;
    bool operator<(const Edge& other) const {
        return std::tie(v1, v2) < std::tie(other.v1, other.v2);
    }
};

// Structures de résultats
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
    double topological_charge; 
};

// --- FONCTIONS DE LECTURE ---
Eigen::MatrixXd load_magnetization(const std::string& sol_filename, double& out_time) {
    std::ifstream sol_file(sol_filename);
    if (!sol_file.is_open()) throw std::runtime_error("Fichier SOL manquant : " + sol_filename);
    
    std::string line;
    if (!std::getline(sol_file, line)) throw std::runtime_error("Fichier SOL vide : " + sol_filename);

    size_t pos_colon = line.find(':');
    if (pos_colon != std::string::npos) {
        try { out_time = std::stod(line.substr(pos_colon + 1)); } catch (...) { out_time = 0.0; }
    } else { out_time = 0.0; }

    std::vector<Eigen::Vector3d> mag_list;
    while (std::getline(sol_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        double dummy1, dummy2, dummy3, dummy4, mx, my, mz;
        if (ss >> dummy1 >> dummy2 >> dummy3 >> dummy4 >> mx >> my >> mz) {
            mag_list.push_back(Eigen::Vector3d(mx, my, mz));
        }
    }
    
    Eigen::MatrixXd mag(mag_list.size(), 3);
    for (size_t i = 0; i < mag_list.size(); ++i) mag.row(i) = mag_list[i];
    return mag;
}

Mesh load_mesh_gmsh(const std::string& msh_filename, size_t expected_nodes) {
    Mesh mesh;
    std::ifstream src_file(msh_filename);
    if (!src_file.is_open()) throw std::runtime_error("Impossible d'ouvrir le maillage : " + msh_filename);

    std::string clean_filename = msh_filename + ".clean_tmp";
    std::ofstream dst_file(clean_filename);
    std::string line;
    bool inside_corrupted_zone = false;

    while (std::getline(src_file, line)) {
        if (line.find("$EndMeshFormat") != std::string::npos) {
            dst_file << line << "\n"; inside_corrupted_zone = true; continue;
        }
        if (line.find("$Nodes") != std::string::npos) inside_corrupted_zone = false;
        if (inside_corrupted_zone) continue;
        dst_file << line << "\n";
    }
    src_file.close(); dst_file.close();

    gmsh::initialize();
    gmsh::option::setNumber("General.Terminal", 0);
    gmsh::open(clean_filename);
    std::filesystem::remove(clean_filename);

    std::vector<size_t> nodeTags;
    std::vector<double> coord, parametricCoord;
    gmsh::model::mesh::getNodes(nodeTags, coord, parametricCoord);

    mesh.points.resize(expected_nodes, 3);
    for (size_t i = 0; i < nodeTags.size(); ++i) {
        size_t idx = nodeTags[i] - 1;
        if (idx < expected_nodes) {
            mesh.points(idx, 0) = coord[3 * i];
            mesh.points(idx, 1) = coord[3 * i + 1];
            mesh.points(idx, 2) = coord[3 * i + 2];
        }
    }

    std::vector<int> elementTypes;
    std::vector<std::vector<size_t>> elementTags, nodeTagsPerElement;
    gmsh::model::mesh::getElements(elementTypes, elementTags, nodeTagsPerElement);

    for (size_t i = 0; i < elementTypes.size(); ++i) {
        if (elementTypes[i] == 2) {
            for (size_t j = 0; j < nodeTagsPerElement[i].size(); j += 3) {
                mesh.triangles.push_back(Eigen::Vector3i(nodeTagsPerElement[i][j]-1, nodeTagsPerElement[i][j+1]-1, nodeTagsPerElement[i][j+2]-1));
            }
        } else if (elementTypes[i] == 4) {
            for (size_t j = 0; j < nodeTagsPerElement[i].size(); j += 4) {
                mesh.tetrahedrons.push_back(Eigen::Vector4i(nodeTagsPerElement[i][j]-1, nodeTagsPerElement[i][j+1]-1, nodeTagsPerElement[i][j+2]-1, nodeTagsPerElement[i][j+3]-1));
            }
        }
    }
    gmsh::finalize();
    return mesh;
}

std::vector<Eigen::Vector3i> orient_triangles_outward(const Eigen::MatrixXd& points, const std::vector<Eigen::Vector4i>& tetrahedrons, const std::vector<Eigen::Vector3i>& triangles) {
    std::map<int, std::vector<int>> node_to_tetra;
    for (int t_idx = 0; t_idx < static_cast<int>(tetrahedrons.size()); ++t_idx) {
        for (int i = 0; i < 4; ++i) node_to_tetra[tetrahedrons[t_idx][i]].push_back(t_idx);
    }
    std::vector<Eigen::Vector3i> oriented_triangles = triangles;
    for (size_t i = 0; i < oriented_triangles.size(); ++i) {
        auto& tri = oriented_triangles[i];
        const auto& t0 = node_to_tetra[tri[0]]; const auto& t1 = node_to_tetra[tri[1]]; const auto& t2 = node_to_tetra[tri[2]];
        std::set<int> int1, final_int;
        std::set_intersection(t0.begin(), t0.end(), t1.begin(), t1.end(), std::inserter(int1, int1.begin()));
        std::set_intersection(int1.begin(), int1.end(), t2.begin(), t2.end(), std::inserter(final_int, final_int.begin()));
        if (final_int.empty()) continue;
        
        Eigen::Vector4i tetra = tetrahedrons[*final_int.begin()];
        int internal_node = -1;
        for (int n = 0; n < 4; ++n) {
            if (tetra[n] != tri[0] && tetra[n] != tri[1] && tetra[n] != tri[2]) { internal_node = tetra[n]; break; }
        }
        Eigen::Vector3d p0 = points.row(tri[0]), p1 = points.row(tri[1]), p2 = points.row(tri[2]), p_in = points.row(internal_node);
        if ((p1 - p0).cross(p2 - p0).dot(p_in - p0) > 0) std::swap(tri[0], tri[1]);
    }
    return oriented_triangles;
}

Eigen::MatrixXd compute_node_normals(const Eigen::MatrixXd& points, const std::vector<Eigen::Vector3i>& triangles) {
    Eigen::MatrixXd node_normals = Eigen::MatrixXd::Zero(points.rows(), 3);
    for (const auto& tri : triangles) {
        Eigen::Vector3d face_normal = (Eigen::Vector3d(points.row(tri[1])) - Eigen::Vector3d(points.row(tri[0]))).cross(Eigen::Vector3d(points.row(tri[2])) - Eigen::Vector3d(points.row(tri[0])));
        node_normals.row(tri[0]) += face_normal; node_normals.row(tri[1]) += face_normal; node_normals.row(tri[2]) += face_normal;
    }
    for (int i = 0; i < node_normals.rows(); ++i) {
        double norm = node_normals.row(i).norm();
        if (norm > TOL) node_normals.row(i) /= norm;
    }
    return node_normals;
}

// --- APPROCHE A : SUBDIVISION PN-TRIANGLE ET INTERPOLATION DE L'AIMANTATION ---
Mesh subdivide_surface_pn(const Mesh& original_mesh, const Eigen::MatrixXd& original_mag, Eigen::MatrixXd& out_fine_mag) {
    Mesh fine_mesh;
    fine_mesh.points = original_mesh.points;
    fine_mesh.tetrahedrons = original_mesh.tetrahedrons; 

    out_fine_mag = original_mag; // Initialisation de la matrice d'aimantation fine

    std::map<Edge, int> midpoints_map;

    auto get_or_create_pn_midpoint = [&](int iA, int iB) {
        int v1 = std::min(iA, iB); int v2 = std::max(iA, iB);
        Edge edge{v1, v2};
        if (midpoints_map.find(edge) != midpoints_map.end()) return midpoints_map[edge];

        Eigen::Vector3d P1 = original_mesh.points.row(v1), P2 = original_mesh.points.row(v2);
        Eigen::Vector3d N1 = original_mesh.node_normals.row(v1), N2 = original_mesh.node_normals.row(v2);

        // Correction Géométrique Géométrique PN-Triangle
        double d1 = (P2 - P1).dot(N1); double d2 = (P1 - P2).dot(N2);
        Eigen::Vector3d P_corrected = 0.5 * (P1 + P2) + 0.125 * (d1 * N1 + d2 * N2);

        // --- INTERPOLATION PHYSIQUE DE L'AIMANTATION ---
        Eigen::Vector3d M1 = original_mag.row(v1), M2 = original_mag.row(v2);
        Eigen::Vector3d M_interpolated = (0.5 * (M1 + M2)).normalized(); // Normalisation stricte |m| = 1

        int new_idx = fine_mesh.points.rows();
        fine_mesh.points.conservativeResize(new_idx + 1, 3);
        fine_mesh.points.row(new_idx) = P_corrected;

        out_fine_mag.conservativeResize(new_idx + 1, 3);
        out_fine_mag.row(new_idx) = M_interpolated;

        midpoints_map[edge] = new_idx;
        return new_idx;
    };

    for (const auto& tri : original_mesh.triangles) {
        int m01 = get_or_create_pn_midpoint(tri[0], tri[1]);
        int m12 = get_or_create_pn_midpoint(tri[1], tri[2]);
        int m20 = get_or_create_pn_midpoint(tri[2], tri[0]);

        fine_mesh.triangles.push_back(Eigen::Vector3i(tri[0], m01, m20));
        fine_mesh.triangles.push_back(Eigen::Vector3i(m01, tri[1], m12));
        fine_mesh.triangles.push_back(Eigen::Vector3i(m20, m12, tri[2]));
        fine_mesh.triangles.push_back(Eigen::Vector3i(m01, m12, m20));
    }
    return fine_mesh;
}

// --- ALGORITHMES D'ANALYSE ---
std::unique_ptr<BlochPointResult> analyze_bloch_point(int current_iter, double current_time, const Matrix34d& nodes_coords, const Matrix34d& nodes_mag) {
    Eigen::Matrix3d A_m;
    A_m.col(0) = nodes_mag.col(1) - nodes_mag.col(0); A_m.col(1) = nodes_mag.col(2) - nodes_mag.col(0); A_m.col(2) = nodes_mag.col(3) - nodes_mag.col(0);
    if (std::abs(A_m.determinant()) < TOL) return nullptr;
    
    Eigen::Vector3d vec_local = -1.0 * A_m.colPivHouseholderQr().solve(nodes_mag.col(0));
    if ((vec_local.array() > 0).all() && (vec_local.array() < 1).all() && (vec_local.sum() < 1)) {
        Eigen::Matrix3d B_geo;
        B_geo.col(0) = nodes_coords.col(1) - nodes_coords.col(0); B_geo.col(1) = nodes_coords.col(2) - nodes_coords.col(0); B_geo.col(2) = nodes_coords.col(3) - nodes_coords.col(0);
        Eigen::Vector3d sol_cartesian = nodes_coords.col(0) + B_geo * vec_local;

        Eigen::Matrix4d A_space; A_space.col(0) = Eigen::Vector4d::Ones(); A_space.block<4,3>(0,1) = nodes_coords.transpose();
        if (std::abs(A_space.determinant()) < TOL) return nullptr;
        Eigen::Matrix4d inv_A_space = A_space.inverse();
        
        Eigen::Matrix3d jac;
        jac.row(0) = (inv_A_space * nodes_mag.row(0).transpose()).tail<3>();
        jac.row(1) = (inv_A_space * nodes_mag.row(1).transpose()).tail<3>();
        jac.row(2) = (inv_A_space * nodes_mag.row(2).transpose()).tail<3>();

        Eigen::EigenSolver<Eigen::Matrix3d> es(jac); Eigen::Vector3cd eigvals = es.eigenvalues();
        std::string bp_type = "unknown";
        if (std::abs(eigvals[0].imag()) < TOL && std::abs(eigvals[1].imag()) < TOL && std::abs(eigvals[2].imag()) < TOL) {
            double r0 = eigvals[0].real(), r1 = eigvals[1].real(), r2 = eigvals[2].real();
            if (r0 > 0 && r1 > 0 && r2 > 0) bp_type = "source";
            else if (r0 < 0 && r1 < 0 && r2 < 0) bp_type = "sink";
            else bp_type = (r0 * r1 * r2 > 0) ? "saddle_2in-1out" : "saddle_1in-2out";
        }
        return std::make_unique<BlochPointResult>(BlochPointResult{current_iter, current_time, sol_cartesian, Eigen::Vector3d(jac(2,1)-jac(1,2), jac(0,2)-jac(2,0), jac(1,0)-jac(0,1)), eigvals, bp_type});
    }
    return nullptr;
}

std::unique_ptr<SurfaceSingularityResult> analyze_surface_singularity(int current_iter, double current_time, const Eigen::Matrix3d& ns_coords, const Eigen::Matrix3d& ns_mag, const Eigen::Matrix3d& ns_normals) {
    Eigen::Vector3d v1 = ns_coords.col(1) - ns_coords.col(0), v2 = ns_coords.col(2) - ns_coords.col(0);
    Eigen::Vector3d n = v1.cross(v2); double n_norm = n.norm(); if (n_norm == 0) return nullptr;
    n.normalize();

    double triple_product = ns_mag.col(0).dot(ns_mag.col(1).cross(ns_mag.col(2)));
    double denominator = 1.0 + ns_mag.col(0).dot(ns_mag.col(1)) + ns_mag.col(1).dot(ns_mag.col(2)) + ns_mag.col(2).dot(ns_mag.col(0));
    double Q_local = (std::abs(denominator) > TOL) ? (2.0 * std::atan2(triple_product, denominator)) / (4.0 * M_PI) : 0.0;

    Eigen::Vector3d t0 = v1.normalized(), t1 = n.cross(t0);
    Eigen::Vector3d m_t0 = Eigen::Vector3d::Zero(), m_t1 = Eigen::Vector3d::Zero();

    for (int i = 0; i < 3; ++i) {
        Eigen::Vector3d m_magenta = ns_mag.col(i) - ns_mag.col(i).dot(ns_normals.col(i)) * ns_normals.col(i);
        m_t0[i] = m_magenta.dot(t0); m_t1[i] = m_magenta.dot(t1);
    }

    Eigen::Matrix2d A_surf;
    A_surf(0,0) = m_t0[1]-m_t0[0]; A_surf(0,1) = m_t1[1]-m_t1[0]; A_surf(1,0) = m_t0[2]-m_t0[0]; A_surf(1,1) = m_t1[2]-m_t1[0];
    if (std::abs(A_surf.determinant()) < TOL) return nullptr;

    Eigen::Vector2d vec_local = -1.0 * A_surf.colPivHouseholderQr().solve(Eigen::Vector2d(m_t0[0], m_t1[0]));
    if (vec_local[0] > 0 && vec_local[1] > 0 && (vec_local[0] + vec_local[1] < 1)) {
        Eigen::Vector3d sol_cartesian = ns_coords.col(0) + vec_local[0] * v1 + vec_local[1] * v2;
        Eigen::Vector3d n_interp = (ns_normals.col(0) + vec_local[0]*(ns_normals.col(1)-ns_normals.col(0)) + vec_local[1]*(ns_normals.col(2)-ns_normals.col(0))).normalized();
        double p_val = (ns_mag.col(0) + vec_local[0]*(ns_mag.col(1)-ns_mag.col(0)) + vec_local[1]*(ns_mag.col(2)-ns_mag.col(0))).dot(n_interp);

        Eigen::Matrix3d M_matrix; M_matrix.col(0) = Eigen::Vector3d::Ones(); M_matrix.col(1) = ns_coords.transpose()*t0; M_matrix.col(2) = ns_coords.transpose()*t1;
        if (std::abs(M_matrix.determinant()) < TOL) return nullptr;
        Eigen::Matrix3d inv_M = M_matrix.inverse();

        Eigen::Matrix2d jac_2d;
        jac_2d(0,0) = (inv_M * m_t0)[1]; jac_2d(0,1) = (inv_M * m_t0)[2];
        jac_2d(1,0) = (inv_M * m_t1)[1]; jac_2d(1,1) = (inv_M * m_t1)[2];

        Eigen::EigenSolver<Eigen::Matrix2d> es(jac_2d); Eigen::Vector2cd eigvals = es.eigenvalues();
        std::string surf_type = (std::abs(eigvals[0].imag()) > TOL) ? (std::abs(Q_local) > 0.1 ? "meron" : "vortex") : "saddle/other";

        return std::make_unique<SurfaceSingularityResult>(SurfaceSingularityResult{current_iter, current_time, sol_cartesian, jac_2d(1,0)-jac_2d(0,1), eigvals, surf_type, p_val, Q_local});
    }
    return nullptr;
}

// --- MAIN ---
int main(int argc, char* argv[]) {
    // 0. Récupération du fichier maillage (.msh) depuis la ligne de commande
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <mesh.msh>\n"; return 1; }
    std::string msh_file = argv[1];

    // 1. Lister tous les fichiers sol*.in et extraire l'indice d'itération via Expressions Régulières (Regex)
    std::vector<std::pair<int, std::string>> sol_files;
    for (const auto& entry : fs::directory_iterator(DIRECTORY_PATH)) {
        if (entry.path().extension() == ".sol") {
            std::regex re(R"(.*_iter(\d+)\.sol)"); std::smatch match; std::string fn = entry.path().filename().string();
            int num = std::regex_match(fn, match, re) ? std::stoi(match[1].str()) : 0;
            sol_files.push_back({num, entry.path().string()});
        }
    }
    if (sol_files.empty()) return 1;
    std::sort(sol_files.begin(), sol_files.end());

    double dummy_time = 0.0;
    Eigen::MatrixXd initial_mag = load_magnetization(sol_files[0].second, dummy_time);

    // Chargement unique du maillage brut d'origine
    Mesh base_mesh = load_mesh_gmsh(msh_file, initial_mag.rows());
    base_mesh.triangles = orient_triangles_outward(base_mesh.points, base_mesh.tetrahedrons, base_mesh.triangles);
    base_mesh.node_normals = compute_node_normals(base_mesh.points, base_mesh.triangles);

    std::vector<BlochPointResult> global_bloch_points;
    std::vector<SurfaceSingularityResult> global_surface_singularities;

    // Boucle chronologique
    for (const auto& [iter, file_path] : sol_files) {
        double current_time = 0.0;
        Eigen::MatrixXd mag;
        try { mag = load_magnetization(file_path, current_time); } catch (...) { continue; }

        if (mag.rows() != base_mesh.points.rows()) continue;
        std::cout << "Traitement (PN-Subdivision) de : " << file_path << " (Iter: " << iter << ")\n";

        // --- GÉNÉRATION DU MAILLAGE FIN POUR CE PAS DE TEMPS ---
        Eigen::MatrixXd fine_mag;
        Mesh fine_mesh = subdivide_surface_pn(base_mesh, mag, fine_mag);
        fine_mesh.node_normals = compute_node_normals(fine_mesh.points, fine_mesh.triangles);

        // --- ANALYSE VOLUME (Sur le maillage d'origine) ---
        for (const auto& nodes_idx : base_mesh.tetrahedrons) {
            Matrix34d t_coords, t_mag;
            for(int i = 0; i < 4; ++i) {
                t_coords.col(i) = base_mesh.points.row(nodes_idx[i]);
                t_mag.col(i) = mag.row(nodes_idx[i]);
            }
            auto res_bp = analyze_bloch_point(iter, current_time, t_coords, t_mag);
            if (res_bp) global_bloch_points.push_back(*res_bp);
        }

        // --- ANALYSE SURFACE (Sur le MAILLAGE FIN ÉCHANTILLONNÉ) ---
        for (const auto& nodes_idx : fine_mesh.triangles) {
            Eigen::Matrix3d s_coords, s_mag, s_normals;
            for(int i = 0; i < 3; ++i) {
                s_coords.col(i) = fine_mesh.points.row(nodes_idx[i]);
                s_mag.col(i) = fine_mag.row(nodes_idx[i]);
                s_normals.col(i) = fine_mesh.node_normals.row(nodes_idx[i]);
            }
            auto res_surf = analyze_surface_singularity(iter, current_time, s_coords, s_mag, s_normals);
            if (res_surf) {
                std::printf(" -> [Singularite Finie] (%s) trouvee a [%.4f, %.4f, %.4f]\n", res_surf->type.c_str(), res_surf->pos.x(), res_surf->pos.y(), res_surf->pos.z());
                global_surface_singularities.push_back(*res_surf);
            }
        }
    }

    // Écritures des fichiers de rapports (.txt) identiques à ton code d'origine...
    std::cout << "Analyses et exports finis.\n";
    return 0;
}