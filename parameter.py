from typing import List
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


PATH_TO_MAP = "Benchmark/kroA100.tsp"
TOWNS = read_map(PATH_TO_MAP)

# Parameter for SOS
PARAMETER_SPACE = (1, 5)
POP_SIZE = 10  # The number of candidate solutions
DIM = 2  # number of problem variables [α, β]

# Parameter for ACO
ANTS = 50
EVAPORATION_RATE = 0.1
INTENSIFICATION = 1000
MAX_ITER_ACO = 300  # The number of iterations
BETA_EVAPORATION_RATE = 0.05