GTTLOB:
pourquoi diviser par le first price bid:

porquoi passer par un graph:

proximité entre les niveaux

lien bid ask

GNN sait pondérer:
un mur de 5000 actions bid3 pese plus que 50 bid1

num_layers = 2:
permet d'atteindre noeud 2 par 2

GATCONV est une brique grapha ttention pour chaque noeud elle regarde ses voisins et attribue un poids avant dans faire la moyenne

concat=False car on fait la moyenne

RELU casse la linéarité

Dropour pour coupé aléatoirement des neuronnes pour pas trop fit sur l'entrainement

edge_index traits (aretes du graphe)

global_add_pool = somme de tous les noeuds d'un snap

renvoit un vect de 64

20 nœuds        arêtes GAT      somme
[price,size] —> Attention —>   Σnœuds   =  [64 nombres]
(side flag)      ×2 couches


positional encoding:

encodage de position car le transformeur connait pas les positions

d_model = taille des vecteurs qui entrent (64)
max_len = nombre maxi de pas de temps que l'on veut supporter

BLOCK 3
Transfo temporel:
g₀  g₁  g₂  …  g₉₉     (chaîne de 100 vecteurs)
│   │   │       │
└─── Transformer ───►   sortie enrichie

Après avoir résumé chaque snapshot du carnet en un vecteur de 64 nombres
(bloc 1) et ajouté l’information d’ordre chronologique (bloc 2),
il nous reste à lire la séquence complète :
Le Transformer est l’équivalent, pour des séries temporelles, de ce qu’est
un réseau de neurones récurrents, mais :

il regarde tous les pas de temps en parallèle (self-attention) ;

il apprend quelles positions sont importantes entre elles
(ex. « le mur de liquidité vu 80 ms plus tôt influence encore le prix »).


Pourquoi ces chiffres ?
Paramètre	Explication simplifiée
d_model = 64	même taille que la sortie du GNN (cohérent).
nhead = 4	le vecteur est découpé en 4 sous-espaces de 16 dims ; chaque “tête” apprend à repérer un type de motif temporel.
ffn = 256	petite MLP interne (×4 d_model), recommandé dans l’article Transformer.
layers = 2	on empile deux “blocs” ; suffisant pour une fenêtre de 100 ticks.
dropout = 0.1	évite l’overfit.

TransformerEncoderLayer =

Self-attention (apprend les dépendances longues)

Add & Norm (résidu)

MLP (dim_feedforward)

Add & Norm à nouveau.

nn.TransformerEncoder = empile num_layers fois cette brique.

Remarques débutant
Option	Pourquoi on la garde
batch_first=False	PyTorch attend (longueur, batch, dim) ; on garde la convention pour ne pas mélanger les axes.
norm_first=True	Place la normalisation avant l’opération ––> flux de gradients plus stable, utile quand on a peu d’étages.
gelu	Fonction d’activation douce, standard dans GPT/BERT ; mieux que ReLU pour les Transformers.

3.2 Comment circule une séquence à l’intérieur
Entrée x : (seq_len, batch, 64)
déjà enrichie des positions (bloc 2).

Self-attention : pour chaque pas de temps, le réseau calcule
« Quels autres pas de temps sont pertinents ? » … 4 fois (4 têtes).

Somme résiduelle (+ normalisation) : facilite l’apprentissage profond.

MLP 256→64 avec GELU : ajoute une transformation non-linéaire.

Répèté 2 couches → permet des dépendances de second ordre.

Au final, on obtient la même taille de séquence, mais chaque vecteur
contient maintenant de l’information venant de tous les autres
instants pertinents.

3.3 Pourquoi c’est mieux qu’un LSTM ici ?
LSTM	Transformer
Lit la séquence pas à pas : difficile de capter un lien entre t et t-80 (vanishing gradient).	Lit tout d’un coup : l’attention relie directement t et t-80.
Difficile à paralléliser sur GPU.	Entièrement parallèle, très rapide.
Sensible à la longueur max (100 vs 200 nécessite re-training souvent).	Le même modèle fonctionne tant qu’on reste < max_len (ici 5000).

3.4 À retenir pour un débutant
Le Transformer sert ici de “radar temporel” : il scrute tout
l’historique pour comprendre la dynamique.

Les paramètres d_model, nhead, layers sont des boutons :
plus tu les montes, plus le modèle est puissant… et lourd à entraîner.

Les réglages (64, 4, 2) sont un point de départ sûr :
ils tiennent dans < 5 M de paramètres et tournent sur GPU 4 Go.

Une fois que tu auras fait un premier run et obtenu un score de référence,
tu pourras jouer sur ces chiffres (par ex. passer à layers=3 si tu
augmentes la fenêtre à 200 ticks).


Objectif : réunir ce que nous avons déjà construit
• un encodeur de graphe (bloc 1)
• un codage de position (bloc 2)
• un Transformer temporel (bloc 3)
et y ajouter une tête de classification pour prédire : down / flat / up.


Pourquoi 3 entrées ? Ce sont nos trois features par nœud.

Pourquoi 64 sorties ? Compromis précision / mémoire.

Pourquoi 2 couches ? Deux sauts suffisent pour que l’info circule
sur nos petits graphes.