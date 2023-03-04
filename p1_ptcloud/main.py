import open3d as o3d
import numpy as np
import laspy
import time
import matplotlib.pyplot as plt

"""
You are using laspy 2.0, which has several improvements over 1.x
            but with several breaking changes.
            To stay on laspy 1.x: `pip install laspy<2.0.0`
            
            In short:
              - To read a file do: las = laspy.read('somefile.laz')
              - To create a new LAS data do: las = laspy.create(point_format=2, file_version='1.2')
              - To write a file previously read or created: las.write('somepath.las')
            See the documentation for more information about the changes https://laspy.readthedocs.io/en/latest/
"""


# Wczytanie chumry punktów w foracielas
def las_to_o3d(file):
    # las_pcd = laspy.file.File(file, mode='r')
    las_pcd = laspy.read(file)
    x = las_pcd.x
    y = las_pcd.y
    z = las_pcd.z

    # Normalizacja koloru
    # r = las_pcd.red/max(las_pcd.red)
    r = las_pcd.intensity
    g = las_pcd.intensity
    b = las_pcd.intensity

    # Konwersja do format NumPy do o3d
    las_points = np.vstack((x, y, z)).transpose()
    las_colors = np.vstack((r, g, b)).transpose()
    chmura_punktow = o3d.geometry.PointCloud()
    chmura_punktow.points = o3d.utility.Vector3dVector(las_points)
    chmura_punktow.colors = o3d.utility.Vector3dVector(las_colors)
    return chmura_punktow


def manual_pt_cloud_cropping(point_cloud):
    print("Manualne przycinanie chmury punktów")
    print("Etapy przetwarzania danych:")
    print(" (0) Manualne zdefiniowanie widoku poprzez obrót myszka lub:")
    print(" (0.1) Podwójne wciśnięcie klawisza X - zdefiniowanie widoku ortogonalnego względem osi X")
    print(" (0.2) Podwójne wciśnięcie klawisza Y - zdefiniowanie widoku ortogonalnego względem osi Y")
    print(" (0.3) Podwójne wciśnięcie klawisza Z - zdefiniowanie widoku ortogonalnego względem osi Z")
    print(" (1) Wciśnięcie klawisza K - zmiana na tryb rysowania")
    print(
        " (2.1) Wybór zaznaczenia poprzez wciśnięcie lewego przycisku myszy i interaktywnego narysowania prostokąta lub"
    )
    print(" (2.2) przytrzymanie przycisku ctrl i wybór wierzchołków poligonu lewym przyciskiem myszy")
    print(" (3) Wciśnięcie klawisza C - wybór zaznaczonego fragmentu chmury punktów i zapis do pliku")
    print(" (4) Wciśnięcie klawisza F - powrót do interaktywnego wyświetlania chmury punktów")
    o3d.visualization.draw_geometries_with_editing([point_cloud], window_name="Przycinanie chmury punktów")


def point_picking(point_cloud):
    print("Pomiar punktów na chmurze punktów")
    print("Etapy pomiaru punktów: ")
    print(" (1.1) Pomiar punktu - shift + lewy przycisk myszy")
    print(" (1.2) Cofniecie ostatniego pomiaru - shift + prawy przycisk myszy")
    print(" (2) Koniec pomiaru - wciśnięcie klawisza Q")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pomiar punktów")
    vis.add_geometry(point_cloud)
    vis.run()  # user picks points
    vis.destroy_window()
    print("Koniec pomiaru")
    print(vis.get_picked_points())  # indeksy
    return vis.get_picked_points()


def get_ptcloud_bbox(point_cloud, type="AxisAlignedBoundingBox"):
    if type == "AxisAlignedBoundingBox":
        print("Obliczanie obszaru opracowania zorientowanego względem osi XYZ")
        aabb = point_cloud.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)
        print("AxisAlignedBoundingBox: ", aabb)
        o3d.visualization.draw_geometries([point_cloud, aabb], window_name="AxisAlignedBoundingBox")
    else:
        print("Obliczanie obszaru opracowania zorientowanego względem chmury punktów")
        obb = point_cloud.get_oriented_bounding_box()
        obb.color = (0, 1, 0)
        print("OrientedBoundingBox", obb)
        o3d.visualization.draw_geometries([point_cloud, obb], window_name="OrientedBoundingBox")


# Filtracja chmur punktów metodą StatisticalOutlierRemoval
def statistical_outlier_removal(chmura_punktów, liczba_sąsiadów=30, std_ratio=2.0):
    chmura_punktów_odfiltrowana, ind = chmura_punktów.remove_statistical_outlier(
        nb_neighbors=liczba_sąsiadów, std_ratio=std_ratio
    )
    start = time.time()
    punkty_odstające = chmura_punktów.select_by_index(ind, invert=True)
    print("Wyświetlanie chmur punktów - punkty odstające (kolor czerwony), chmura i punktów (kolor RGB): ")
    punkty_odstające.paint_uniform_color([1, 0, 0])
    end = time.time()
    computation_time = end - start
    print("Outlier removal time:\t", computation_time, "s")
    o3d.visualization.draw_geometries([chmura_punktów_odfiltrowana, punkty_odstające])
    return chmura_punktów_odfiltrowana, punkty_odstające, computation_time


# Filtracja chmur punktów metodą radius_outlier_removal
def radius_outlier_removal(chmura_punktów, min_liczba_punktow=30, promień_sfery=2.0):
    start = time.time()
    chmura_punktow_odfiltrowana, ind = chmura_punktów.remove_radius_outlier(nb_points=min_liczba_punktow, radius=0.05)
    punkty_odstające = chmura_punktów.select_by_index(ind, invert=True)
    print("Wyświetlanie chmur punktów - punkty odstające (kolor czerwony), chmura punktów(kolor RGB)")
    punkty_odstające.paint_uniform_color([1, 0, 0])
    end = time.time()
    computation_time = end - start
    print("Outlier removal time:\t", computation_time, "s")
    o3d.visualization.draw_geometries([chmura_punktow_odfiltrowana, punkty_odstające])
    return chmura_punktow_odfiltrowana, punkty_odstające, computation_time


# Downsampling chmur punktów
# Regularyzacja chmury punktów
def voxel_downsample(chmura_punktów, odległość_pomiedzy_wokselami=0.1):
    chmura_punktów_woksele = chmura_punktów.voxel_down_sample(voxel_size=odległość_pomiedzy_wokselami)
    print("Wyświetlanie chmury punktów w regularnej siatce wokseli - odległość %f: " % odległość_pomiedzy_wokselami)
    o3d.visualization.draw_geometries([chmura_punktów_woksele])
    return chmura_punktów_woksele


def uniform_down_sample(chmura_punktów, co_n_ty_punkt=10):
    # usuwanie_co_n_tego_punktu_z_chmury_punktów
    chmura_punktów_co_n_ty = chmura_punktów.uniform_down_sample(every_k_points=co_n_ty_punkt)
    print("Wyświetlanie chmura punktów zredukowanej co %i: " % co_n_ty_punkt)
    o3d.visualization.draw_geometries([chmura_punktów_co_n_ty])
    return chmura_punktów_co_n_ty


# Klasteryzacja punktów algorytmem DBSCAN
def dbscan_clustering(chmura_punktów, odleglosc_miedzy_punktami, min_punktow, progress=True):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        klasy = np.array(
            chmura_punktów.cluster_dbscan(
                eps=odleglosc_miedzy_punktami, min_points=min_punktow, print_progress=progress
            )
        )
    liczba_klas = klasy.max() + 1
    print("Algorytm DBSCAN wykrył %i klas" % liczba_klas)
    colors = plt.get_cmap("tab20")(klasy / (klasy.max() if klasy.max() > 0 else 1))
    colors[klasy < 0] = 0
    chmura_punktów.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([chmura_punktów])
    return chmura_punktów


def main():
    # filepath = r"F:\311512\FTP\p2\cloud.las"
    filepath = r"C:\SEM6\FTP2\p1_ptcloud\cloud.las"
    chmura_punktow = las_to_o3d(filepath)
    o3d.io.write_point_cloud("test.pcd", chmura_punktow)
    print(chmura_punktow)
    manual_pt_cloud_cropping(chmura_punktow)
    point_picking(chmura_punktow)
    get_ptcloud_bbox(chmura_punktow)
    print("\nOutlier Removal\n")
    statistical_outlier_removal(chmura_punktow)  # fast
    #  radius_outlier_removal(chmura_punktow) # slow
    print("\nDownsampling chmury punktów\n")
    voxel_downsample(chmura_punktow)
    uniform_down_sample(chmura_punktow)
    print("\nDBSCAN Density-based spatial clustering of applications with noise\n")
    print(type(chmura_punktow))
    # dbscan_clustering(chmura_punktow, odleglosc_miedzy_punktami=0.05, min_punktow=10) # slow


if __name__ == "__main__":
    main()
    # ZROBIC DO PUNKTU 5 WŁĄCZNIE DO DBSCAN
