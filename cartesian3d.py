"""Create a 3d cartesian graph."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

MATRIX_UNITS_VECTORS = [1, 0, 0], [0, 1, 0], [0, 0, 1]


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


class Vector():
    """Create a 3d vector item."""

    def __init__(self, units_vectors, coordinates, name=''):
        """Construct the vector object."""
        x, y, z = coordinates
        self.units_vectors = units_vectors
        self.coordinates_untrans = coordinates
        self.coordinates = self.__get_true_coordinate(coordinates)
        self.components = self.__get_components(coordinates)
        self.name = name
        self.determinent = self.__get_determinent()
        self.norme = self.__get_norme()
        self.drawn = False

    def __get_components(self, coordinates):
        """Return components of the vector in the space."""
        components_list = [[], [], []]
        for i, unit_verctor in enumerate(self.units_vectors):
            for coord in range(3):
                components_list[i].append(coordinates[i] * unit_verctor[coord])
        return components_list

    def __get_norme(self):
        """Return the norme of the vector in the space."""
        array = np.array(self.components)
        norme = round(linalg.norm(array), 3)
        return norme

    def __get_true_coordinate(self, coordinates):
        """Return coordinates of the vector in the space."""
        new_coordinates = [0, 0, 0]
        for i, unit_verctor in enumerate(self.units_vectors):
            for j, coord in enumerate(unit_verctor):
                new_coordinates[j] += coordinates[i] * coord
        return new_coordinates

    def __get_determinent(self):
        array = np.array(self.components)
        determinent = round(linalg.det(array),3)
        return determinent


class Cartesian():
    """Create a 3d cartesian space."""

    def __init__(self, trans_matrix='' , draw_unit=1):
        """Cronstruct the object."""
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111, projection='3d')
        self.max_coordinates = [[0, 0], [0, 0], [0, 0]]
        self.vector_list = {}
        self.additions_list = {}

        if trans_matrix == '':
            matrix_units_vectors = MATRIX_UNITS_VECTORS
        else:
            matrix_units_vectors = trans_matrix

        if draw_unit == 1:
            self.__draw_units(matrix_units_vectors)

        self.matrix_units_vectors = matrix_units_vectors

    def __draw_units(self, matrix_units_vectors):
        """Draw unit vvector and determinent."""
        self.__draw_unit_determinent(matrix_units_vectors)
        self.__draw_units_vector(matrix_units_vectors)

    def __update_size(self, unit_list=[], vector_list=[]):
        """Update the size of the cartesian space."""
        all_vectors = []
        for vector in unit_list:
            all_vectors.append(vector)

        for vector in vector_list:
            if vector.drawn == True:
                all_vectors.append(vector.coordinates)
                for i in range(3):
                    all_vectors.append(vector.components[i])

        for vector in all_vectors:
            plus = max(self.max_coordinates[1])
            minus = min(self.max_coordinates[0])
            for i, coordinate in enumerate(vector):
                if coordinate <= minus:
                    minus = coordinate - 1
                if coordinate >= plus:
                    plus = coordinate + 1
            for i in range(3):
                if abs(plus) > abs(minus):
                    minus = -abs(plus)
                else:
                    plus = abs(minus)
                self.max_coordinates[i][0] = minus
                self.max_coordinates[i][1] = plus
        self.__draw_axis(self.max_coordinates)

    def __draw_unit_determinent(self, matrix_units_vectors):
        """Add the unit deteminent."""
        X, Y, Z = self.__get_deteminent_array(matrix_units_vectors)
        self.ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.07)

    def __draw_determinent(self, matrix_vector):
        """Add the deteminent."""
        X, Y, Z = self.__get_deteminent_array(matrix_vector)
        self.ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.07)

    def __get_deteminent_array(self, matrix_vectors):
        """Get determinent as numpy array to pass in plot_surface method."""
        i = matrix_vectors[0]
        j = matrix_vectors[1]
        k = matrix_vectors[2]

        x = np.array([
            [0, i[0], i[0]+j[0], j[0], 0],
            [k[0], i[0]+k[0], i[0]+k[0]+j[0], k[0]+j[0], k[0]],
            [0, i[0], i[0]+k[0], k[0], 0],
            [j[0], i[0]+j[0], i[0]+k[0]+j[0], k[0]+j[0], j[0]]
            ])

        y = np.array([
            [0, i[1], i[1]+j[1], j[1], 0],
            [k[1], i[1]+k[1], i[1]+k[1]+j[1], k[1]+j[1], k[1]],
            [0, i[1], i[1]+k[1], k[1], 0],
            [0+j[1], i[1]+j[1], i[1]+k[1]+j[1], k[1]+j[1], 0+j[1]]
            ])

        z = np.array([
            [0, i[2], i[2]+j[2], j[2], 0],
            [k[2], i[2]+k[2], i[2]+k[2]+j[2], k[2]+j[2], k[2]],
            [0, i[2], i[2]+k[2], k[2], 0],
            [j[2], i[2]+j[2], i[2]+k[2]+j[2], k[2]+j[2], j[2]]
            ])

        return x, y, z

    def __draw_axis(self, size):
        """Add axis and draw the doted axis."""
        """Draw the axis."""
        self.ax1.set_xlabel('X')
        self.ax1.set_xlim(size[0])
        self.ax1.set_ylabel('Y')
        self.ax1.set_ylim(size[1])
        self.ax1.set_zlabel('Z')
        self.ax1.set_zlim(size[2])

        """Draw the dotes axis."""
        arrow_prop_dict = dict(
            mutation_scale=10,
            arrowstyle='-',
            linestyle='--',
            shrinkA=0,
            shrinkB=0,
            alpha=0.2
            )
        a = Arrow3D(size[0], [0, 0], [0, 0], **arrow_prop_dict)
        self.ax1.add_artist(a)
        a = Arrow3D([0, 0], size[1], [0, 0], **arrow_prop_dict)
        self.ax1.add_artist(a)
        a = Arrow3D([0, 0], [0, 0], size[2], **arrow_prop_dict)
        self.ax1.add_artist(a)

    def __label_axis(self, size):
        """Label the axis."""
        self.ax1.text(0.0, 0.0, -0.3, r'$0$')
        self.ax1.text(size[0][1], 0, 0, r'$X$')
        self.ax1.text(0, size[1][1], 0, r'$Y$')
        self.ax1.text(0, 0, size[2][1], r'$Z$')

    def __draw_units_vector(self, matrix_units_vectors):
        """Draw units vectors."""
        arrow_prop_dict = dict(
            mutation_scale=20,
            arrowstyle='->',
            shrinkA=0,
            shrinkB=0
            )

        self.__draw_components_vectors(
            matrix_units_vectors,
            arrow_prop_dict,
            unit=True
            )

        """Label unit vectors."""
        self.ax1.text(0.0, 0.0, -0.3, r'$0$')
        self.ax1.text(0.7, 0, -0.2, r'$i$', color='r')
        self.ax1.text(-0.2, 0.7, 0, r'$j$', color='b')
        self.ax1.text(0, -0.2, 0.7, r'$k$', color='g')

    def __draw_components_vectors(self, vector_matrix, arrow_prop_dict, unit=0):
        """Draw units vectors."""
        arrow_prop_dict = dict(
            mutation_scale=20,
            arrowstyle='->',
            shrinkA=0,
            shrinkB=0
            )

        if unit is True:
            color = ['r', 'b', 'g']
        else:
            color = ['k', 'k', 'k']
            arrow_prop_dict['linestyle'] = ':'
            arrow_prop_dict['alpha'] = 0.7

        a = ArrowFromOrigin(
            vector_matrix[0],
            **arrow_prop_dict,
            color=color[0]
            )
        self.ax1.add_artist(a)
        a = ArrowFromOrigin(
            vector_matrix[1],
            **arrow_prop_dict,
            color=color[1]
            )
        self.ax1.add_artist(a)
        a = ArrowFromOrigin(
            vector_matrix[2],
            **arrow_prop_dict,
            color=color[2]
            )
        self.ax1.add_artist(a)

    def new_vector(self, vector_coordinates, name=''):
        """Add a vector object."""
        if name == '':
            name = 'V{}'.format(len(self.vector_list)+1)
        else:
            pass

        vector = Vector(
            units_vectors=self.matrix_units_vectors,
            coordinates=vector_coordinates,
            name=name
            )
        self.vector_list[name] = vector
        return vector

    def draw_vector(self, vector_name, color='k', det=False, comp=False,
                    fade=False, added=False, no_name=False):
        """Draw units vectors."""
        vector = ''
        alpha = True
        linestyle = '-'
        arrowstyle ='-|>'

        try:
            vector = self.vector_list[vector_name]
        except:
            raise EnvironmentError

        if fade is True:
            alpha = 0.5
            linestyle = '-.'
            arrowstyle = '-|>'

        arrow_prop_dict = dict(
            mutation_scale=20,
            arrowstyle=arrowstyle,
            shrinkA=0,
            shrinkB=0,
            color=color,
            linestyle=linestyle,
            alpha=alpha
            )
        a = ArrowFromOrigin(vector.coordinates, **arrow_prop_dict)
        self.ax1.add_artist(a)

        x, y, z = vector.coordinates
        name_n_coord = vector.name + ' ' + str(vector.coordinates)
        if no_name is True:
            pass
        else:
            self.ax1.text(x, y, z, name_n_coord)

        if added is True:
            self.draw_added_vector(vector)
        if comp is True:
            self.__draw_components_vectors(vector.components, arrow_prop_dict)
        if det is True:
            self.__draw_determinent(vector.components)
        self.vector_list[vector_name].drawn = True

    def draw_added_vector(self, vector):
        """Draw component for vector addition."""
        components = []
        try:
            components = self.additions_list[vector.name]
        except:
            raise EnvironmentError

        v1 = self.vector_list[components[0]]
        v2 = self.vector_list[components[1]]

        arrow_prop_dict = dict(
            mutation_scale=20,
            arrowstyle='->',
            linestyle='--',
            shrinkA=0,
            shrinkB=0,
            color='k',
            alpha=0.7
            )
        a = Arrow3D(
                [v1.coordinates[0], v1.coordinates[0] + v2.coordinates[0]],
                [v1.coordinates[1], v1.coordinates[1] + v2.coordinates[1]],
                [v1.coordinates[2], v1.coordinates[2] + v2.coordinates[2]],
                **arrow_prop_dict
                 )
        if v1.drawn is False:
            if v2.drawn is True:
                self.draw_vector(v2.name, no_name=True, fade=True, color='k')
                a = Arrow3D(
                    [v2.coordinates[0], v1.coordinates[0] + v2.coordinates[0]],
                    [v2.coordinates[1], v1.coordinates[1] + v2.coordinates[1]],
                    [v2.coordinates[2], v1.coordinates[2] + v2.coordinates[2]],
                    **arrow_prop_dict
                    )
            else:
                self.draw_vector(v1.name, no_name=True, fade=True, color='k')

        self.ax1.add_artist(a)

    def add_vector(self, v1, v2, name=''):
        """Add 2 vector to create a third vector."""
        v1_coordinates = []
        v2_coordinates = []
        v3_coordinates = []

        try:
            v1_coordinates = self.vector_list[v1].coordinates_untrans
            v2_coordinates = self.vector_list[v2].coordinates_untrans
        except:
            raise EnvironmentError

        for value in range(len(v1_coordinates)):
            v3_coordinates.append(
                v1_coordinates[value] + v2_coordinates[value])
        if name == '':
            name = '{} + {}'.format(
                self.vector_list[v1].name,
                self.vector_list[v2].name)
        else:
            name = name
        self.new_vector(v3_coordinates, name=name)

        self.additions_list[name] = [v1, v2]

    def _print_vector_info(self):
        """Print vector info in the terminal."""
        print('''
----------------------------------------
----------------------------------------
                RESOLUTION
----------------------------------------
----------------------------------------
Units vectors :
    î'  =        {}
    ĵ'  =        {}
    k'  =        {}
        '''.format(
            self.matrix_units_vectors[0], 
            self.matrix_units_vectors[1], 
            self.matrix_units_vectors[2])
            )

        for vector in self.vector_list.values():
            vector_info = [
                vector.name,
                vector.coordinates,
                vector.norme,
                vector.components,
                vector.determinent]

            print('''
----------------------------------------
Vector name         =   {}
Vector coodinates   =   {}
Vector norme        =   {}
Vector components î =   {}
Vector components ĵ =   {}
Vector components k =   {}
Vector determinent  =   {}
-----------------------------------------
        '''.format(
                vector_info[0],
                vector_info[1],
                vector_info[2],
                vector_info[3][0],
                vector_info[3][1],
                vector_info[3][2],
                vector_info[4])
                )

    def show(self):
        """Show the cartesian space in maplotlib."""
        self.__update_size(self.matrix_units_vectors, self.vector_list.values())
        self.__label_axis(self.max_coordinates)
        self._print_vector_info()
        plt.show()
        plt.close(self.fig)


def main():
    """Test function."""
    transformation = [-1, -2, 3], [-1, 4, 0], [5, -2, 1]
    cart = Cartesian(trans_matrix=transformation)
    cart.draw_vector([2, 3, 2])
    cart.draw_vector([-1, 3, -2])
    cart.draw_vector([2, -1, 1])
    cart.show()


if __name__ == '__main__':
    pass
