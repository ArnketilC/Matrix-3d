# -*- Coding:UTF-8 -*-
"""Petit soft pour faire des graphique cartésiens en 3d."""

from cartesian3d import Cartesian as repere


def main():
    """Entrer le code ici."""

    """Exemple de transformations."""
    # transformation = [1, 0, 2], [-1, -1, 0], [0, 2, 1]
    transformation = [1, 0, 0], [0, 1, 0], [0, 0, 1]

    """Création du repere."""
    cart = repere(trans_matrix=transformation)

    """Exemple de vecteurs."""
    cart.new_vector([2, 2, 2])
    cart.new_vector([-1, 2, -2])
    # cart.draw_vector('V1', comp=True, det=True)
    # cart.draw_vector('V2', fade=False)

    # cart.draw_vector('V1', fade=True)
    # cart.draw_vector('V2', fade=True)

    """Exemple addition vectorielle."""
    cart.add_vector('V1', 'V2')
    cart.draw_vector('V1 + V2', added=True)

    cart.show()


if __name__ == '__main__':
    main()
