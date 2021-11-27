from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LightSource
from moviepy.video.io.bindings import mplfig_to_npimage
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl.mesh import Mesh

RGBA = Tuple[float, float, float, float]


class RandomProjection(object):
    def __init__(
            self,
            azimuth: int, altitude: int,
            darkest_shadow_surface: RGBA, brightest_lit_surface: RGBA
    ):
        self.__azimuth = azimuth
        self.__altitude = altitude

        self.__dk = np.array(darkest_shadow_surface)
        self.__lt = np.array(brightest_lit_surface)

    def __call__(self, mesh: Mesh) -> np.ndarray:
        random_rotation_vectors = 2 * (np.random.rand(3) - 0.5)
        random_rotation_angle = float(np.radians(360 * np.random.rand()))
        mesh.rotate(random_rotation_vectors, random_rotation_angle)

        poly_mesh = self.__create_illumination(mesh)
        array_img = RandomProjection.__plot_to_array_data(mesh, poly_mesh)

        return array_img

    def __create_illumination(self, mesh: Mesh) -> Poly3DCollection:
        def shade(s):
            return (self.__lt - self.__dk) * s + self.__dk

        poly_mesh = mplot3d.art3d.Poly3DCollection(mesh.vectors)

        ls = LightSource(azdeg=self.__azimuth, altdeg=self.__altitude)
        sns = ls.shade_normals(mesh.get_unit_normals(), fraction=1.0)
        rgba = np.array([shade(s) for s in sns])

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
