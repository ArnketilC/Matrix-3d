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
    cart.draw_vector([2, 2, 2], det=True, comp=True)
    cart.draw_vector([-1, 2, -2], det=True, comp=True)
    cart.draw_vector([2, -1, 1], det=True, comp=True)

    cart.show()


if __name__ == '__main__':
    main()
