# -*- Coding:UTF-8 -*-
"""Petit soft pour faire des graphique cartésiens en 3d."""

from cartesian3d import Cartesian as repere


def main():
    """Entrer le code ici."""
    """Exemple de transformations."""
    transformation = [1, 0, 2], [-1, -1, 0], [0, 2, 1]
    # transformation = [1, 0, 0], [0, 1, 0], [0, 0, 1]

    """Création du repere."""
    cart = repere(trans_matrix=transformation)

    """Exemple de vecteurs."""
    cart.new_vector([2, 2, -3])
    cart.new_vector([-3, -1, -2])
    # cart.new_vector([2, 2, 0])
    # cart.new_vector([0, 0, -1])

    """ Addition de vecteurs."""
    cart.add_vector('V1', 'V2', 'V1,2')
    cart.add_vector('V1', 'V1,2')

    """Dessine les vecteurs."""
    # cart.draw_vector('V1', comp=True, det=True)
    # cart.draw_vector('V2', comp=True, det=True)
    cart.draw_vector('V1')
    cart.draw_vector('V2', fade=True)
    cart.draw_vector('V1,2', fade=True)
    cart.draw_vector('V1 + V1,2', det=True, comp=True, added=True, color='c')

    cart.show()


if __name__ == '__main__':
    main()
