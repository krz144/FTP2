import open3d as o3d
import numpy as np
import laspy
import matplotlib.pyplot as plt
import copy


# Wczytanie chumry punktów w foracielas
def las_to_o3d(file):
    # las_pcd = laspy.file.File(file, mode="r")
    las_pcd = laspy.read(file)
    x = las_pcd.x
    y = las_pcd.y
    z = las_pcd.z

    # Normalizacja koloru
    # r = las_pcd.red/max(las_pcd.red)
    r = las_pcd.red / max(las_pcd.intensity)
    g = las_pcd.green / max(las_pcd.intensity)
    b = las_pcd.blue / max(las_pcd.intensity)

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
    """
    Removes points that are further away from their neighbors compared to the average for the point cloud. It takes two input parameters:

    * nb_neighbors: specifies how many neighbors are taken into account in order to calculate the average distance for a given point.
    * std_ratio:  allows setting the threshold level based on the standard deviation of the average distances across the point cloud. The lower this number the more aggressive the filter will be.

    # http://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html
    """
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
    # http://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html
    chmura_punktow_odfiltrowana, ind = chmura_punktów.remove_radius_outlier(
        nb_points=min_liczba_punktow, radius=promień_sfery
    )
    punkty_odstające = chmura_punktów.select_by_index(ind, invert=True)
    print("Wyświetlanie chmur punktów - punkty odstające (kolor czerwony), chmura punktów(kolor RGB)")
    punkty_odstające.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([chmura_punktow_odfiltrowana, punkty_odstające])
    return chmura_punktow_odfiltrowana, punkty_odstające


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


# =============================================== ZAJECIA 3 P3 ===================


# Generowanie modelu TIN metodą Ball Pivoting
def wyznaczanie_normalnych(chmura_punktów):  # dziala git
    print("Wyznaczanie normalnych dla punktów w chmurze punktów")
    chmura_punktów.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    # Jeżeli istnieją normalne to są zerowane
    chmura_punktów.estimate_normals()
    return chmura_punktów


def ball_pivoting(
    chmura_punktow, promienie_kul=[0.005, 0.01, 0.03, 0.9, 1.5], savepath="model_3d.ply"
):  #  0.005, 0.01, 0.03, 0.9, 1.5]
    chmura_punktow_z_normalnymi = wyznaczanie_normalnych(chmura_punktow)
    print("Przetwarzanie danych algorytmem Ball Pivoting")
    tin = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        chmura_punktow_z_normalnymi, o3d.utility.DoubleVector(promienie_kul)
    )
    o3d.visualization.draw_geometries([chmura_punktow_z_normalnymi, tin])
    o3d.io.write_triangle_mesh(savepath, tin)  # trzeba zmienic promienie kul by caly model powstal


# Generowanie modelu metodą Poissona
def wyznaczanie_normalnych(chmura_punktów):  # dziala git
    print("Wyznaczanie normalnych dla punktów w chmurze punktów")
    chmura_punktów.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    # Jeżeli istnieją normalne to są zerowane
    chmura_punktów.estimate_normals()
    return chmura_punktów


def poisson(chmura_punktów):
    print("Przetwarzanie danych algorytmem Poissona")
    wyznaczanie_normalnych(chmura_punktów)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        tin, gęstość = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(chmura_punktów, depth=15)
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


def poisson_filtracja(chmura_punktów):
    tin, gęstość = poisson(chmura_punktów)
    wyświetlanie_gęstości(gęstość, tin)
    filtracja_modelu_w_oparciu_o_gęstość(tin, gęstość, kwantyl=0.05)
    o3d.io.write_triangle_mesh("model_3d.ply", tin)


# #Wykrywanie punktów metoda ISS
# def detektor_ISS(chmura_punktów):
#     print('Detekcja punktów charakterystycznych')
#     keypoints = o3d.geometry.keypoint.compute_iss_keypoints(chmura_punktów)
#     keypoints.paint_uniform_color([1.0, 0.0, 0.0])
#     o3d.visualization.draw_geometries([keypoints])

# #Poprawa wizualizacji wykrytych punktów charakterystycznych
# def wizualizacja_punktow_charakterystycznych_za_pomoca_sfer(keypoints):
#     sfery = o3d.geometry.TriangleMesh()
#     for keypoint in keypoints.points:
#         sfery = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
#         sfery.translate(keypoint)
#         sfery += sfery
#     sfery.paint_uniform_color([1.0, 0, 0.0])
#     return sfery


# ============================== p4 orientacja chur punktow ===========================
# Orientacja metodą Target-based
def wyswietalnie_par_chmur_punktow(chmura_referencyjna, chmura_orientowana, transformacja):
    ori_temp = copy.deepcopy(chmura_orientowana)
    ref_temp = copy.deepcopy(chmura_referencyjna)
    ori_temp.paint_uniform_color([1, 0, 0])
    ref_temp.paint_uniform_color([0, 1, 0])
    ori_temp.transform(transformacja)
    o3d.visualization.draw_geometries([ori_temp, ref_temp])


def orientacja_target_based(chmura_referencyjna, chmura_orientowana, typ="Pomiar", Debug=False):
    print("Orientacja chmur punktów metoda Target based")
    wyswietalnie_par_chmur_punktow(chmura_referencyjna, chmura_orientowana, np.identity(4))
    if typ == "Pomiar":
        print("Pomierz min. 3 punkty na chmurze referencyjnej: ")
        pkt_ref = point_picking(chmura_referencyjna)
        print("Pomierz min. 3 punkty orientowanej ")
        pkt_ori = point_picking(chmura_orientowana)
        assert len(pkt_ref) >= 3 and len(pkt_ori) >= 3
        assert len(pkt_ref) == len(pkt_ori)
    elif typ == "Plik":
        print("Wyznaczenia parametrów transformacji na podstawie punktów pozyskanych z plików tekstowych")
        # Wczytanie chmur punktów w postaci plików tekstowych  # TODO
        # Przygotowanie plików ref i ori
    else:  # Inna metoda
        print("Wyznaczenie parametrów na podstawie analizy deskryptorów")
        # Analiza deskryptorów
    corr = np.zeros((len(pkt_ori), 2))
    corr[:, 0] = pkt_ori
    corr[:, 1] = pkt_ref
    print(corr)
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans = p2p.compute_transformation(chmura_orientowana, chmura_referencyjna, o3d.utility.Vector2iVector(corr))
    # ^ tu zmienialismy kolejnosc c_refi c_orient...
    if Debug:
        print(trans)
        wyswietalnie_par_chmur_punktow(chmura_referencyjna, chmura_orientowana, trans)
        o3d.io.write_point_cloud("pc_after_orientation.pcd", chmura_orientowana)
    # analiza_statystyczna(chmura_referencyjna, chmura_orientowana,trans)  # TODO
    return trans  # macierz translacji


# Dopasowanie deskryptorów
def dopasowanie_deskryptorów(
    chmura_orientowana, chmura_referencyjna, orientowany_fpfh, referencyjny_fpfh, max_odleglość
):
    print("Dopasowanie deskryptorów")
    param_orientacji = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        chmura_orientowana,
        chmura_referencyjna,
        orientowany_fpfh,
        referencyjny_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            liczba_iteracji=100, maximum_correspondence_distance=max_odleglość
        ),
    )
    return param_orientacji


# Dopasowanie deskryptorów metoda RANSAC
def dopasowanie_deskryptorów_RANSAC(
    chmura_orientowana, chmura_referencyjna, orientowany_fpfh, referencyjny_fpfh, max_odleglość
):
    param_orientacji = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        chmura_orientowana,
        chmura_referencyjna,
        orientowany_fpfh,
        referencyjny_fpfh,
        True,
        max_odleglość,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_odleglość),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return param_orientacji


# Orientacja chmur punktów metodami ICP
def ICP_registration(source, target, threshold=1.0, trans_init=np.identity(4), metoda="p2p"):
    print("Analiza dokładności wstępnej orientacji")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)
    if metoda == "p2p":
        print("Orientacja ICP <Punkt do punktu>")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        print(reg_p2p)
        print("Macierz transformacji:")
        print(reg_p2p.transformation)
        wyswietalnie_par_chmur_punktow(target, source, reg_p2p.transformation)
        information_reg_p2p = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, threshold, reg_p2p.transformation
        )
        return reg_p2p.transformation, information_reg_p2p
    elif metoda == "p2pl":
        print("Wyznaczanie normalnych")
        source.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # Jeżeli istnieją normalne to są zerowane
        source.estimate_normals()
        target.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # Jeżeli istnieją normalne to są zerowane
        target.estimate_normals()
        print("Orientacja ICP <Punkt do płaszczyzny>")
        reg_p2pl = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        print(reg_p2pl)
        print("Macierz transformacji:")
        print(reg_p2l.transformation)
        wyswietalnie_par_chmur_punktow(source, target, reg_p2l.transformation)
        information_reg_p2pl = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, threshold, reg_p2pl.transformation
        )
        return reg_p2l.transformation, information_reg_p2pl
    elif metoda == "cicp":
        reg_cicp = o3d.pipelines.registration.registration_colored_icp(source, target, threshold, trans_init)
        print(reg_cicp)
        print("Macierz transformacji:")
        print(reg_cicp.transformation)
        wyswietalnie_par_chmur_punktow(source, target, reg_cicp.transformation)
        information_reg_cicp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, threshold, reg_cicp.transformation
        )
        return reg_cicp.transformation, information_reg_cicp
    else:
        print("Nie wybrano odpowiedniego sposobu transformacji")


def main():
    # mamy dwie chmury punktów w formacie las
    chmura_dji = r"C:\SEM6\FTP2\p4_pt_cloud_katedra\chmura_dji_phantom.las"
    chmura_zdjecia = r"C:\SEM6\FTP2\p4_pt_cloud_katedra\chmura_zdjecia_naziemne.las"

    # wczytanie do formatu o3d
    pc_ref = las_to_o3d(chmura_dji)
    pc_orientowana = las_to_o3d(chmura_zdjecia)

    # wstępne odszumienie (filtracja) chmury punktów powstałej ze zdjęć naziemnych
    pc_orientowana, _ = statistical_outlier_removal(chmura_punktów=pc_orientowana, liczba_sąsiadów=1000, std_ratio=0.7)
    # pc_orientowana, _ = statistical_outlier_removal(chmura_punktów=pc_orientowana, liczba_sąsiadów=30, std_ratio=0.7)
    o3d.visualization.draw_geometries([pc_orientowana])

    # wzajemna orientacja chmur punktów
    translation_matrix = orientacja_target_based(pc_ref, pc_orientowana, typ="Pomiar", Debug=False)
    pc_orientowana.transform(translation_matrix)

    # połączenie chmur punktów
    pc_joined = pc_orientowana + pc_ref
    o3d.io.write_point_cloud("pc_joined.pcd", pc_joined)

    # filtracja chmury punktów
    chmura_punktów_odfiltrowana, _ = statistical_outlier_removal(pc_joined, liczba_sąsiadów=50, std_ratio=0.5)

    # wygenerowanie modelu 3d, ball_pivoting, eksport do formatu ply
    model_3d = ball_pivoting(
        pc_joined,
        promienie_kul=[0.005, 0.01, 0.03, 0.9, 1.5],
        savepath=r"C:\SEM6\FTP2\p4_pt_cloud_katedra\model_katedra.ply",
    )


if __name__ == "__main__":
    main()
