import open3d as o3d
import numpy as np
import laspy
import matplotlib.pyplot as plt


# Wczytanie chumry punktów w foracielas
def las_to_o3d(file):
    # las_pcd = laspy.file.File(file, mode="r")
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
    punkty_odstające = chmura_punktów.select_by_index(ind, invert=True)
    print("Wyświetlanie chmur punktów - punkty odstające (kolor czerwony), chmura i punktów (kolor RGB): ")
    punkty_odstające.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([chmura_punktów_odfiltrowana, punkty_odstające])
    return chmura_punktów_odfiltrowana, punkty_odstające


# Filtracja chmur punktów metodą radius_outlier_removal
def radius_outlier_removal(chmura_punktów, min_liczba_punktow=30, promień_sfery=2.0):
    chmura_punktow_odfiltrowana, ind = chmura_punktów.remove_radius_outlier(nb_points=min_liczba_punktow, radius=0.05)
    punkty_odstające = chmura_punktów.select_by_index(ind, invert=True)
    print("Wyświetlanie chmur punktów - punkty odstające (kolor czerwony), chmura punktów(kolor RGB)")
    punkty_odstające.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([chmura_punktow_odfiltrowana, punkty_odstające])
    return chmura_punktow_odfiltrowana, punkty_odstające


# =============================================== ZAJECIA 3 P3 ===================


# Generowanie modelu TIN metodą Ball Pivoting
def wyznaczanie_normalnych(chmura_punktów):  # dziala git
    print("Wyznaczanie normalnych dla punktów w chmurze punktów")
    chmura_punktów.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    # Jeżeli istnieją normalne to są zerowane
    chmura_punktów.estimate_normals()
    return chmura_punktów


def ball_pivoting(chmura_punktow, promienie_kul=[0.005, 0.01, 0.03, 0.9, 1.5]):  #  0.005, 0.01, 0.03, 0.9, 1.5]
    chmura_punktow_z_normalnymi = wyznaczanie_normalnych(chmura_punktow)
    print("Przetwarzanie danych algorytmem Ball Pivoting")
    tin = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        chmura_punktow_z_normalnymi, o3d.utility.DoubleVector(promienie_kul)
    )
    o3d.visualization.draw_geometries([chmura_punktow_z_normalnymi, tin])
    # o3d.io.write_triangle_mesh("model_3d.ply", tin)  # trzeba zmienic promienie kul by caly model powstal


# Generowanie modelu metodą Poissona
def wyznaczanie_normalnych(chmura_punktów):  # dziala git
    print("Wyznaczanie normalnych dla punktów w chmurze punktów")
    chmura_punktów.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    # Jeżeli istnieją normalne to są zerowane
    chmura_punktów.estimate_normals()
    return chmura_punktów


def poisson(chmura_punktów, depth):
    print("Przetwarzanie danych algorytmem Poissona")
    wyznaczanie_normalnych(chmura_punktów)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        tin, gęstość = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(chmura_punktów, depth=depth)
    print(tin)
    o3d.visualization.draw_geometries([tin])
    return tin, gęstość


# Wyświetlanie gęstości chmur punktów
def wyświetlanie_gęstości(gęstość, tin):
    gęstość = np.asarray(gęstość)
    gęstość_colors = plt.get_cmap("plasma")((gęstość - gęstość.min()) / (gęstość.max() - gęstość.min()))
    gęstość_colors = gęstość_colors[:, :3]
    gęstość_mesh = o3d.geometry.TriangleMesh()
    gęstość_mesh.vertices = tin.vertices
    gęstość_mesh.triangles = tin.triangles
    gęstość_mesh.triangle_normals = tin.triangle_normals
    gęstość_mesh.vertex_colors = o3d.utility.Vector3dVector(gęstość_colors)
    o3d.visualization.draw_geometries([gęstość_mesh])


# Usunięcie punktów w oparciu o wyliczona gęstość
def filtracja_modelu_w_oparciu_o_gęstość(tin, gęstość, kwantyl=0.01):
    print("Usuniecie trójkątów powstałych w oparciu o małą liczbę punktów")
    vertices_to_remove = gęstość < np.quantile(gęstość, kwantyl)
    tin.remove_vertices_by_mask(vertices_to_remove)
    print(tin)
    o3d.visualization.draw_geometries([tin])


def poisson_filtracja(chmura_punktów, depth):
    tin, gęstość = poisson(chmura_punktów, depth=depth)
    wyświetlanie_gęstości(gęstość, tin)
    filtracja_modelu_w_oparciu_o_gęstość(tin, gęstość, kwantyl=0.05)
    o3d.io.write_triangle_mesh("model_3d.ply", tin)


def main():
    filepath = r"F:\311512\FTP\p3\cloud.las"
    filepath = r"C:\SEM6\FTP2\p1_ptcloud\cloud.las"
    filepath_mc = r"F:\311512\FTP\p3\minicloud.las"
    filepath_mc = r"C:\SEM6\FTP2\p2_tin\minicloud.las"
    chmura_punktow = las_to_o3d(filepath)
    minicloud = las_to_o3d(filepath_mc)
    print("AAAAAAAAAA" * 10)
    # wyznaczanie_normalnych(minicloud)
    # print(np.asarray(minicloud.normals))
    # o3d.visualization.draw_geometries([minicloud])

    # ball_pivoting(minicloud)
    # poisson_filtracja(minicloud, depth=15) # co to ma być?


if __name__ == "__main__":
    main()
    # p3 zaczynamy od punktu6.2
