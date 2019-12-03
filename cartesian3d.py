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
    """Create a 3d cartesian area."""

    def __init__(self, ax1, matrix_units_vector):
        """Cronstruct the object."""
        super(Cartesian, self).__init__()
        self.matrix_units_vector = matrix_units_vector
        self.size = [[-1,2],[-1,2],[-1,2]]
        # for vector in units_vector:
        #     temp = [vector[0] - 1, vector[1] + 1]
        #     self.size.append(temp)
        self.ax1 = ax1
        self.draw_unit_determinent(self.ax1)
        self.draw_axis(self.ax1)
        self.draw_units_vector(self.ax1)

    def draw_unit_determinent(self, ax1):
        """Add the unit deteminent."""
        size = [1, 1, 1]
        center = [0.5, 0.5, 0.5]
        X, Y, Z = cuboid_data(center, size)
        ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.1)

    def draw_axis(self, ax1):
        """Add axis and draw the doted axis."""
        """Draw the axis."""
        ax1.set_xlabel('X')
        ax1.set_xlim(self.size[0])
        ax1.set_ylabel('Y')
        ax1.set_ylim(self.size[1])
        ax1.set_zlabel('Z')
        ax1.set_zlim(self.size[2])

        """Draw the dotes axis."""
        arrow_prop_dict = dict(mutation_scale=10, arrowstyle='-', linestyle='--', shrinkA=0, shrinkB=0, alpha=0.5)
        a = Arrow3D(self.size[0], [0, 0], [0, 0], **arrow_prop_dict)
        ax1.add_artist(a)
        a = Arrow3D([0, 0], self.size[1], [0, 0], **arrow_prop_dict)
        ax1.add_artist(a)
        a = Arrow3D([0, 0], [0, 0], self.size[2], **arrow_prop_dict)
        ax1.add_artist(a)

        """Label the axis."""
        ax1.text(0.0, 0.0, -0.3, r'$0$')
        ax1.text(2.1, 0, 0, r'$X$')
        ax1.text(0, 2.1, 0, r'$Y$')
        ax1.text(0, 0, 2.1, r'$Z$')

    def draw_units_vector(self, ax1):
        """Draw units vectors."""
        arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
        a = Arrow3D(self.matrix_units_vector[0][0], self.matrix_units_vector[0][1], self.matrix_units_vector[0][2], **arrow_prop_dict, color='r')
        ax1.add_artist(a)
        a = Arrow3D(self.matrix_units_vector[1][0], self.matrix_units_vector[1][1], self.matrix_units_vector[1][2], **arrow_prop_dict, color='b')
        ax1.add_artist(a)
        a = Arrow3D(self.matrix_units_vector[2][0], self.matrix_units_vector[2][1], self.matrix_units_vector[2][2], **arrow_prop_dict, color='g')
        ax1.add_artist(a)

        """Label unit vectors."""
        # delta_size = abs(self.size[0][0]) + abs(self.size[0][1])
        ax1.text(0.0, 0.0, -0.3, r'$0$')
        ax1.text(0.7, 0, -0.2, r'$i$', color='r')
        ax1.text(-0.2, 0.7, 0, r'$j$', color='b')
        ax1.text(0, -0.2, 0.7, r'$k$', color='g')


if __name__ == '__main__':
    length = 1
    width = 1
    height = 1
    matrix_units_vector = [[0, 1], [0, 0], [0, 0]], [[0, 0], [0, 1], [0, 0]], [[0, 0], [0, 0], [0, 1]]
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    Cartesian(ax1, matrix_units_vector)

    plt.show()
