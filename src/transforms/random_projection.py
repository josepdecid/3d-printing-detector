from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from moviepy.video.io.bindings import mplfig_to_npimage
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl.mesh import Mesh


class RandomProjection(object):
    """
    Starting from a 3D STL-versioned file, it creates a random 2D projection of the figure from an arbitrary PoV.
    It also randomizes illumination parameters related to the azimuth and altitude values (int values 0-255).
    And it does the same for the bright and dark surface vectors (RGBA vectors).

    STLMesh -> NPArray
    """

    def __call__(self, mesh: Mesh) -> np.ndarray:
        random_rotation_vectors = 2 * (np.random.rand(3) - 0.5)
        random_rotation_angle = float(np.radians(360 * np.random.rand()))
        mesh.rotate(random_rotation_vectors, random_rotation_angle)

        poly_mesh = RandomProjection.__create_illumination(mesh)
        array_img = RandomProjection.__plot_to_array_data(mesh, poly_mesh)

        return array_img

    @staticmethod
    def __create_illumination(mesh: Mesh) -> Poly3DCollection:
        lt, dk = RandomProjection.__generate_random_brightness_parameters()
        azimuth = float(np.random.rand())
        altitude = float(np.random.rand())

        poly_mesh = mplot3d.art3d.Poly3DCollection(mesh.vectors)

        ls = LightSource(azimuth, altitude)
        sns = ls.shade_normals(mesh.get_unit_normals(), fraction=1.0)
        rgba = np.array([(lt - dk) * s + dk for s in sns])

        poly_mesh.set_facecolor(rgba)

        return poly_mesh

    @staticmethod
    def __plot_to_array_data(mesh: Mesh, poly_mesh: Poly3DCollection) -> np.ndarray:
        figure = plt.figure()
        axes = mplot3d.Axes3D(figure)
        axes.add_collection3d(poly_mesh)

        points = mesh.points.reshape(-1, 3)
        points_top = max(np.ptp(points, 0)) / 2

        controls = [(min(points[:, i]) + max(points[:, i])) / 2 for i in range(3)]
        limits = [[controls[i] - points_top, controls[i] + points_top] for i in range(3)]
        axes.auto_scale_xyz(*limits)
        axes.axis('off')

        np_img = mplfig_to_npimage(figure)
        plt.close(figure)

        return np_img

    @staticmethod
    def __generate_random_brightness_parameters() -> Tuple[np.ndarray, np.ndarray]:
        # TODO: Implement
        pass
