from typing import List, Tuple
import numpy as np

def local_optimize(best_HC: np.ndarray, best_cost: float, distances:np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """
    This local optimization strategy is used to improve the solution quality searched by SOS–ACO and accelerate
    the convergence rate.
    Idea: Given a TSP instance of size n, an HC is denoted as HC = (v1, v2, . . . , vn), where vi is one of the n cities and i represents
    the position of city vi in HC. There are two steps for improving an HC using the local optimization strategy.\n
            Step 1: For each city vk in an HC, find the city vk' which is nearest to vk in the following cities. As shown in Eq. (9), dis(vi
    , vj) is the distance between vi and vj
            Step 2: Reverse the path between vk and vk' so that vk is adjacent to vk'. If the adjusted path is shorter than the original path, the adjusted path is preserved. Otherwise, the original path
    is maintained.
    """
    if not best_HC.size or not distances.size:
        raise ValueError("best_HC or distance matrix cannot be empty")
    if len(best_HC) != len(distances) + 1 or best_HC[0] != best_HC[-1]:
        raise ValueError("best_HC must be a valid Hamiltonian Cycle with the first city repeated at the end")
    
    best_HC = best_HC[:-1]
    n = best_HC.shape[0]
    
    if distances.shape != (n, n):
        raise ValueError("Distance matrix must be square and match the number of cities in best_HC")
    
    for k in range(n-2):
        candidates = best_HC[k + 1:]    
        distance_local = distances[best_HC[k], candidates]
        min_idx = np.argmin(distance_local)
        # d_min = distance_local[min_idx]
        k_ = k + 1 + min_idx
        flag = False
        
        if k_ != k + 1:
            reverse_hc = reverse_path(best_HC, k, k_)
            best_cost_r = cal_cost_HC(reverse_hc, distances)
            
            if best_cost_r < best_cost:
                best_HC = reverse_hc
                best_cost = best_cost_r
                flag = True
    
    if not flag:
        best_HC = np.append(best_HC, best_HC[0])
        return best_HC, best_cost, True
        
    best_HC = np.append(best_HC, best_HC[0])
    return best_HC, best_cost, False


def cal_cost_HC(new_HC: np.ndarray, distances: np.ndarray) -> float:
    """
    Calculate the score of new path base on distance cost in distance_matrix

    :param new_HC: The new path obtained by reverse operation
    :param distances: The matrix of distance costs between each pairs of cities
    """
    # dis = 0
    # for i in range(len(new_HC)-1):
    #     dis += distances[new_HC[i], new_HC[i+1]]
    # dis += distances[new_HC[-1], new_HC[0]]
    # return dis
    
    indices = np.arange(len(new_HC)-1)
    dis = np.sum(distances[new_HC[indices], new_HC[indices+1]]) + distances[new_HC[-1], new_HC[0]]
    return dis
    
def reverse_path(HC: np.ndarray, k: int, k_: int) -> np.ndarray:
    reverse = HC.copy()
    reverse[k+1:k_+1] = reverse[k+1:k_+1][::-1]
    return reverse