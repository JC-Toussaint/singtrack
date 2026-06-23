// Paramètres (en nm)
r  = 40;   // rayon
L  = 1000;   // longueur
lc = 4;    // taille caractéristique

// --- Géométrie 2D : disque (approximation par arcs) ---
Point(1) = { 0, 0, -L/2, lc};     // centre
Point(2) = { r, 0, -L/2, lc};
Point(3) = { 0, r, -L/2, lc};
Point(4) = {-r, 0, -L/2, lc};
Point(5) = { 0,-r, -L/2, lc};

// 4 arcs de cercle
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

Line Loop(10) = {1,2,3,4};
Plane Surface(20) = {10};

// --- Extrusion pour créer le cylindre ---
Extrude {0, 0, L} {
  Surface{20};
}

Physical Volume(300) = {1};
Physical Surface(200) = {41, 37, 33, 29};
Physical Surface(201) = {20};
Physical Surface(200) = {42};

// --- Maillage ---
//Mesh.CharacteristicLengthMin = lc;
//Mesh.CharacteristicLengthMax = lc;

Mesh.Algorithm3D = 4; // Frontal (Netgen), souvent meilleur pour la régularité
//Mesh.Algorithm3D = 10; // HXT

Mesh 3;