// Paramètres du cube (en mètres pour rester cohérent avec la physique)
L = 100; 
lc = 5; // Taille des mailles (plus elle est petite, plus le maillage est fin)
lc = 100;

// Points du cube (centré en 0)
Point(1) = {-L/2, -L/2, -L/2, lc};
Point(2) = { L/2, -L/2, -L/2, lc};
Point(3) = { L/2,  L/2, -L/2, lc};
Point(4) = {-L/2,  L/2, -L/2, lc};
Point(5) = {-L/2, -L/2,  L/2, lc};
Point(6) = { L/2, -L/2,  L/2, lc};
Point(7) = { L/2,  L/2,  L/2, lc};
Point(8) = {-L/2,  L/2,  L/2, lc};

// Lignes
Line(1) = {1, 2}; Line(2) = {2, 3}; Line(3) = {3, 4}; Line(4) = {4, 1};
Line(5) = {5, 6}; Line(6) = {6, 7}; Line(7) = {7, 8}; Line(8) = {8, 5};
Line(9) = {1, 5}; Line(10) = {2, 6}; Line(11) = {3, 7}; Line(12) = {4, 8};

// Surfaces
Line Loop(1) = {1, 2, 3, 4};       Plane Surface(1) = {1}; // Dessous
Line Loop(2) = {5, 6, 7, 8};       Plane Surface(2) = {2}; // Dessus
Line Loop(3) = {1, 10, -5, -9};    Plane Surface(3) = {3}; // Devant
Line Loop(4) = {2, 11, -6, -10};   Plane Surface(4) = {4}; // Droite
Line Loop(5) = {3, 12, -7, -11};   Plane Surface(5) = {5}; // Fond
Line Loop(6) = {4, 9, -8, -12};    Plane Surface(6) = {6}; // Gauche

// Volume
Surface Loop(1) = {2, 3, 1, 4, 5, 6};
Volume(1) = {1};

Physical Surface("top", 13) = {2};
Physical Surface("bottom", 14) = {1};
Physical Surface("lateral", 15) = {6, 5, 4, 3};
Physical Volume("volume", 16) = {1};
