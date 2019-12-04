"""Create a 3d cartesian graph."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    """Create a 3d Arrow object."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        """Construct the arrow object."""
        super(Arrow3D, self).__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """Draw the arrow in the vector space."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super(Arrow3D, self).draw(renderer)


class ArrowFromOrigin(Arrow3D):
    """Create a 3d Arrow from origin object."""

    def __init__(self, coordinates, *args, **kwargs):
        """Construct the arrow from origin object."""
        xs = [0, coordinates[0]]
        ys = [0, coordinates[1]]
        zs = [0, coordinates[2]]
        super(ArrowFromOrigin, self).__init__(xs, ys, zs, *args, **kwargs)

# class determinent():
#     """Check the determinent of a matrix"""
#     def __init__(self, matrix):

def cuboid_data(center, size):
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = np.array([[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]])  # x coordinate of points in inside surface
    y = np.array([[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]])    # y coordinate of points in inside surface
    z = np.array([[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]])                # z coordinate of points in inside surface
    return x, y, z


class Cartesian():
    """Create a 3d cartesian space."""

    def __init__(self, ax1, matrix_units_vector):
        """Cronstruct the object."""
        self.ax1 = ax1
        self.max_coordinates = [[0, 0], [0, 0], [0, 0]]
        self._update_size(matrix_units_vector)
        self._draw_unit_determinent()
        self._draw_units_vector(matrix_units_vector)

    def _update_size(self, vector_list):
        """Update the size of the cartesian space."""
        for vector in vector_list:
            plus = max(self.max_coordinates[1])
            minus = min(self.max_coordinates[0])
            for i, coordinate in enumerate(vector):
                if coordinate <= minus:
                    minus = coordinate - 1
                if coordinate >= plus:
                    plus = coordinate + 1
            for i in range(3):         
                self.max_coordinates[i][0] = minus
                self.max_coordinates[i][1] = plus
        self._draw_axis(self.max_coordinates)

    def _draw_unit_determinent(self):
        """Add the unit deteminent."""
        size = [1, 1, 1]
        center = [0.5, 0.5, 0.5]
        X, Y, Z = cuboid_data(center, size)
        self.ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.1)

    def _draw_axis(self, size):
        """Add axis and draw the doted axis."""
        """Draw the axis."""
        self.ax1.set_xlabel('X')
        self.ax1.set_xlim(size[0])
        self.ax1.set_ylabel('Y')
        self.ax1.set_ylim(size[1])
        self.ax1.set_zlabel('Z')
        self.ax1.set_zlim(size[2])

        """Draw the dotes axis."""
        arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-', linestyle='--', shrinkA=0, shrinkB=0, alpha=0.3)
        a = Arrow3D(size[0], [0, 0], [0, 0], **arrow_prop_dict)
        self.ax1.add_artist(a)
        a = Arrow3D([0, 0], size[1], [0, 0], **arrow_prop_dict)
        self.ax1.add_artist(a)
        a = Arrow3D([0, 0], [0, 0], size[2], **arrow_prop_dict)
        self.ax1.add_artist(a)

        """Label the axis."""
        self.ax1.text(0.0, 0.0, -0.3, r'$0$')
        self.ax1.text(size[0][1], 0, 0, r'$X$')
        self.ax1.text(0, size[1][1], 0, r'$Y$')
        self.ax1.text(0, 0, size[2][1], r'$Z$')

    def _draw_units_vector(self, matrix_units_vector):
        """Draw units vectors."""
        arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
        self.draw_components_vectors(matrix_units_vector, arrow_prop_dict)

        """Label unit vectors."""
        # delta_size = abs(self.size[0][0]) + abs(self.size[0][1])
        self.ax1.text(0.0, 0.0, -0.3, r'$0$')
        self.ax1.text(0.7, 0, -0.2, r'$i$', color='r')
        self.ax1.text(-0.2, 0.7, 0, r'$j$', color='b')
        self.ax1.text(0, -0.2, 0.7, r'$k$', color='g')

    def draw_components_vectors(self, vector_matrix, arrow_prop_dict):
        """Draw units vectors."""
        a = ArrowFromOrigin(vector_matrix[0], **arrow_prop_dict, color='r')
        self.ax1.add_artist(a)
        a = ArrowFromOrigin(vector_matrix[1], **arrow_prop_dict, color='b')
        self.ax1.add_artist(a)
        a = ArrowFromOrigin(vector_matrix[2], **arrow_prop_dict, color='g')
        self.ax1.add_artist(a)

    def draw_vectors(self, vector_cooridnates, names=''):
        """Draw units vectors."""
        arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
        a = ArrowFromOrigin(vector_cooridnates, **arrow_prop_dict)
        self.ax1.add_artist(a)
        # name
        x, y ,z = vector_cooridnates
        self.ax1.text(x, y, z, r'$V1$')
        self._update_size([vector_cooridnates])


if __name__ == '__main__':
    matrix_units_vector = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    cart = Cartesian(ax1, matrix_units_vector)
    cart.draw_vectors([5, 10, 7])
    cart.draw_vectors([-2, 5, -3])

    plt.show()
    plt.close(fig)
