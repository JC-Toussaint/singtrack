#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <set>
#include <algorithm>
#include <memory>
#include <filesystem>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace fs = std::filesystem;

// Aliases pratiques pour les structures à taille fixe
using Matrix34d = Eigen::Matrix<double, 3, 4>;

// --- CONFIGURATION & PARAMÈTRES ---
const std::string MSH_FILE = "cube.msh";
const std::string SOL_FILE = "sol.in";
const double TIME_NS = 0.0;
const double TOL = 1e-16;

// Structures de données pour remplacer meshio
struct Mesh {
    Eigen::MatrixXd points;                  // (N, 3)
    std::vector<Eigen::Vector4i> tetrahedrons; // Liste des indices (4 nœuds)
    std::vector<Eigen::Vector3i> triangles;    // Liste des indices (3 nœuds)
};

// --- CHARGEMENT DES DONNÉES ---
std::pair<Mesh, Eigen::MatrixXd> load_mesh_and_magnetization(const std::string& msh_filename, const std::string& sol_filename) {
    Mesh mesh;
    Eigen::MatrixXd mag;

    // 1. Lecture de l'aimantation
    std::ifstream sol_file(sol_filename);
    if (!sol_file.is_open()) throw std::runtime_error("Fichier SOL manquant : " + sol_filename);
    
    std::vector<Eigen::Vector3d> mag_list;
    double mx, my, mz;
    while (sol_file >> mx >> my >> mz) {
        mag_list.push_back(Eigen::Vector3d(mx, my, mz));
    }
    sol_file.close();

    // 2. Lecture du maillage MSH
    std::ifstream msh_file(msh_filename);
    if (!msh_file.is_open()) throw std::runtime_error("Fichier MSH manquant : " + msh_filename);

    std::string line;
    std::vector<Eigen::Vector3d> points_list;

    while (std::getline(msh_file, line)) {
        if (line.find("$Nodes") != std::string::npos) {
            size_t num_nodes;
            msh_file >> num_nodes;
            points_list.resize(num_nodes);
            for (size_t i = 0; i < num_nodes; ++i) {
                size_t id;
                double x, y, z;
                msh_file >> id >> x >> y >> z;
                points_list[id - 1] = Eigen::Vector3d(x, y, z);
            }
        }
        else if (line.find("$Elements") != std::string::npos) {
            size_t num_elements;
            msh_file >> num_elements;
            for (size_t i = 0; i < num_elements; ++i) {
                size_t id, elem_type, num_tags;
                msh_file >> id >> elem_type >> num_tags;
                for (size_t t = 0; t < num_tags; ++t) { size_t dummy; msh_file >> dummy; }
                
                if (elem_type == 2) { // Triangle
                    int n1, n2, n3;
                    msh_file >> n1 >> n2 >> n3;
                    mesh.triangles.push_back(Eigen::Vector3i(n1 - 1, n2 - 1, n3 - 1));
                } else if (elem_type == 4) { // Tétraèdre
                    int n1, n2, n3, n4;
                    msh_file >> n1 >> n2 >> n3 >> n4;
                    mesh.tetrahedrons.push_back(Eigen::Vector4i(n1 - 1, n2 - 1, n3 - 1, n4 - 1));
                } else {
                    std::getline(msh_file, line);
                }
            }
        }
    }
    msh_file.close();

    mesh.points.resize(points_list.size(), 3);
    mag.resize(mag_list.size(), 3);
    for(size_t i=0; i<points_list.size(); ++i) {
        mesh.points.row(i) = points_list[i];
        mag.row(i) = mag_list[i];
    }

    if (mesh.tetrahedrons.empty()) throw std::runtime_error("Aucun élément tétraédrique trouvé.");
    if (mesh.points.rows() != mag.rows()) throw std::runtime_error("Incohérence Nœuds vs Magnétisation.");

    return {mesh, mag};
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

struct BlochPointResult {
    double time;
    Eigen::Vector3d pos;
    Eigen::Vector3d curl;
    Eigen::Vector3cd eigvals;
    std::string type;
};

struct SurfaceSingularityResult {
    double time;
    Eigen::Vector3d pos;
    double curl_n;
    Eigen::Vector2cd eigvals;
    std::string type;
    double polarity;
};

// --- ANALYSE POINT DE BLOCH (VOLUME) ---
std::unique_ptr<BlochPointResult> analyze_bloch_point(const Matrix34d& nodes_coords, const Matrix34d& nodes_mag) {
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
            TIME_NS, sol_cartesian, Eigen::Vector3d(curl_x, curl_y, curl_z), eigvals, bp_type
        });
    }
    return nullptr;
}

// --- ANALYSE SINGULARITÉ DE SURFACE ---
std::unique_ptr<SurfaceSingularityResult> analyze_surface_singularity(const Eigen::Matrix3d& ns_coords, const Eigen::Matrix3d& ns_mag) {
    // ns_coords et ns_mag ont leurs nœuds en COLONNES (3x3)
    Eigen::Vector3d v1 = ns_coords.col(1) - ns_coords.col(0);
    Eigen::Vector3d v2 = ns_coords.col(2) - ns_coords.col(0);
    Eigen::Vector3d n = v1.cross(v2);
    double n_norm = n.norm();
    if (n_norm == 0) return nullptr;
    n.normalize();

    Eigen::Vector3d t0 = v1.normalized();
    Eigen::Vector3d t1 = n.cross(t0);

    // Projections (noeuds en colonnes -> transpose pour avoir l'équivalent Python)
    Eigen::Vector3d m_t0 = ns_mag.transpose() * t0;
    Eigen::Vector3d m_t1 = ns_mag.transpose() * t1;
    Eigen::Vector3d m_n  = ns_mag.transpose() * n;

    Eigen::Vector3d c_t0 = ns_coords.transpose() * t0;
    Eigen::Vector3d c_t1 = ns_coords.transpose() * t1;

    // Ajustement de la Transposition NumPy (.T) :
    // En Python, les lignes deviennent des colonnes.
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
//        double polarity = (p_val > 0) ? 1.0 : ((p_val < 0) ? -1.0 : 0.0);
        double polarity = static_cast<double>(p_val > 0.0) - static_cast<double>(p_val < 0.0);

        // Correction de M_matrix : hstack en Python sur des vecteurs colonnes (3x1)
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
            TIME_NS, sol_cartesian, curl_n, eigvals, surf_type, polarity
        });
    }
    return nullptr;
}

// --- MAIN ---
int main() {
    std::cout << "Chargement de " << MSH_FILE << " et " << SOL_FILE << "...\n";
    
    Mesh mesh;
    Eigen::MatrixXd mag;
    try {
        std::tie(mesh, mag) = load_mesh_and_magnetization(MSH_FILE, SOL_FILE);
    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << "\n";
        return 1;
    }

    mesh.triangles = orient_triangles_outward(mesh.points, mesh.tetrahedrons, mesh.triangles);

    std::vector<BlochPointResult> bloch_points_list;
    std::vector<SurfaceSingularityResult> surface_sings_list;

    // 1. ANALYSE VOLUME
    std::cout << "Analyse de " << mesh.tetrahedrons.size() << " tétraèdres (Volume)...\n";
    for (const auto& nodes_idx : mesh.tetrahedrons) {
        Matrix34d t_coords, t_mag; 
        for(int i=0; i<4; ++i) {
            t_coords.col(i) = mesh.points.row(nodes_idx[i]);
            t_mag.col(i) = mag.row(nodes_idx[i]);
        }

        auto sign_check = [](const Matrix34d& m, int axis) {
            bool has_pos = false, has_neg = false;
            for(int i=0; i<4; ++i) {
                if(m(axis, i) > 0) has_pos = true;
                if(m(axis, i) < 0) has_neg = true;
            }
            return has_pos && has_neg;
        };

        if (sign_check(t_mag, 0) && sign_check(t_mag, 1) && sign_check(t_mag, 2)) {
            auto res_bp = analyze_bloch_point(t_coords, t_mag);
            if (res_bp) {
                std::printf(" -> [BP Volume] détecté à : [%.5f, %.5f, %.5f] nm (%s)\n", 
                            res_bp->pos.x(), res_bp->pos.y(), res_bp->pos.z(), res_bp->type.c_str());
                bloch_points_list.push_back(*res_bp);
            }
        }
    }

    // 2. ANALYSE SURFACE
    if (!mesh.triangles.empty()) {
        std::cout << "Analyse de " << mesh.triangles.size() << " triangles (Surface)...\n";
        for (const auto& nodes_idx : mesh.triangles) {
            Eigen::Matrix3d s_coords, s_mag; 
            for(int i=0; i<3; ++i) {
                s_coords.col(i) = mesh.points.row(nodes_idx[i]);
                s_mag.col(i) = mag.row(nodes_idx[i]);
            }

            auto res_surf = analyze_surface_singularity(s_coords, s_mag);
            if (res_surf) {
                std::printf(" -> [Singularité Surface] (%s) à : [%.5f, %.5f, %.5f] nm | Polarité: %.1f\n",
                            res_surf->type.c_str(), res_surf->pos.x(), res_surf->pos.y(), res_surf->pos.z(), res_surf->polarity);
                surface_sings_list.push_back(*res_surf);
            }
        }
    }

    // --- SAUVEGARDE ---
    std::ofstream f_vol("volume_bloch_points.txt");
    f_vol << "time(ns)\tx\ty\tz\ttype\n";
    for(const auto& bp : bloch_points_list) {
        f_vol << bp.time << "\t" << bp.pos.x() << "\t" << bp.pos.y() << "\t" << bp.pos.z() << "\t" << bp.type << "\n";
    }

    std::ofstream f_surf("surface_singularities.txt");
    f_surf << "time(ns)\tx\ty\tz\ttype\tpolarity\n";
    for(const auto& ss : surface_sings_list) {
        f_surf << ss.time << "\t" << ss.pos.x() << "\t" << ss.pos.y() << "\t" << ss.pos.z() << "\t" << ss.type << "\t" << ss.polarity << "\n";
    }

    std::cout << "\nTraitements terminés. Fichiers exportés avec succès.\n";
    return 0;
}
