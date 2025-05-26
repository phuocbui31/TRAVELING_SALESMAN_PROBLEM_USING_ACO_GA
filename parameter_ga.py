import numpy as np

def read_map(file_path:str) -> np.ndarray:
    """
    Read all point coordianates from data file

    :param file_path: Path to tsp file.
    :return: NumPy array of shape (n, 2) containing coordinates [x, y] of n cities.
    """
    cities = []
    city_flag = False
    with open(file_path, "r") as file:
        for line in file:
            if line.strip() == "NODE_COORD_SECTION":
                city_flag = True
                continue
            if line.strip() == "EOF":
                break
            if city_flag:
                city_coor = [float(i) for i in line.strip().split()[1:]]
                cities.append(city_coor)

    return np.array(cities, dtype=float)

PATH_TO_MAP = "Benchmark/eil51.tsp"
TOWNS = read_map(PATH_TO_MAP)