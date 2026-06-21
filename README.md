# Singularity Analyzer (Gmsh Multifile)

Ce projet est un outil d'analyse topologique 3D et 2D pour les simulations micromagnétiques. Il permet de détecter, classifier et suivre au cours du temps les singularités magnétiques volumiques (**Points de Bloch**) ainsi que les défauts topologiques de surface (**Vortex de surface, Cross, Merons**, etc.) à partir de fichiers de solutions d'aimantation (`*.sol`) et d'un maillage géométrique au format Gmsh (`*.msh`).

Le code est optimisé pour les calculs géométriques et l'algèbre linéaire grâce à la bibliothèque **Eigen**, et s'appuie sur l'API **Gmsh** pour le traitement des éléments finis.

---

## 🚀 Fonctionnalités

- **Analyse Volumique (3D) :** - Détection du point d'annulation de l'aimantation ($\vec{m} = \vec{0}$) au sein d'éléments tétraédriques par interpolation affine.
  - Calcul du rotationnel local $\vec{\nabla} \times \vec{m}$ au cœur de la singularité.
  - Classification mathématique des points de Bloch par analyse spectrale (Jacobien) : *source, sink, saddle, spiral source, spiral sink, spiral saddle*.

- **Analyse Surfacique (2D) :**
  - Réorientation automatique et robuste des facettes triangulaires pour garantir des normales pointant vers l'extérieur.
  - Projection du champ d'aimantation dans le plan tangent local.
  - Détection et classification des singularités bidimensionnelles (*vortex, anti-vortex/saddle*).
  - Évaluation de la composante normale du rotationnel et de la **polarité** ($\pm 1$).

- **Traitement Temporel & Performance :**
  - Scan automatique et tri chronologique de plusieurs fichiers solutions (`magn_iter*.sol`).
  - Pré-filtrage topologique rapide (*sign-check*) pour sauter les éléments ne pouvant pas contenir de singularités sans calculs matriciels lourds.
  - Chronométrage précis (via `<chrono>`) du traitement géométrique et de la boucle temporelle.

---

## 🛠️ Prérequis & Dépendances

Pour compiler et exécuter ce projet, vous devez disposer des éléments suivants :

1. **Compilateur C++** supportant la norme **C++17** (GCC 8+, Clang 7+, ou MSVC 2019+).
2. **Eigen 3** : Bibliothèque d'algèbre linéaire (en-têtes uniquement).
3. **Gmsh** : Bibliothèque et API de maillage (fichiers d'en-têtes et bibliothèque partagée `libgmsh`).
4. **CMake** (recommandé pour configurer la compilation).

---

## 📦 Compilation

Voici un exemple de configuration standard en ligne de commande avec CMake :

```bash
mkdir build
cd build
cmake ..
make