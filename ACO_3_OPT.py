from Local_Optimize import *
import time
import numpy as np
from parameter import *
import lib
from typing import Tuple
from Local_Optimize import local_optimize
from three_opt import three_opt
import matplotlib.pyplot as plt

class ACO:
    def __init__(self, ants: int, evaporation_rate: float, intensification: float,
                 alpha: float, beta: float, beta_evaporation_rate: float) -> None:
        
        """
        Ant colony optimizer. Finds the path that either maximizes or minimizes distance traveled between nodes. 
        This optimizer is devoted to solving Traveling Salesman Problem (TSP).

        :param ants: Number of ants to traverse the graph.
        :param evaporation_rate: Rate at which pheromone evaporates.
        :param intensification: Constant value added to the best path.
        :param alpha: Weight of pheromone.
        :param beta: Weight of heuristic (1/distance).
        :param beta_evaporation_rate: Rate at which beta decays.
        """
        
        self.ants = ants
        self.evaporation_rate = evaporation_rate
        self.Q = intensification
        self.heuristic_alpha = alpha
        self.heuristic_beta = beta
        self.beta_evaporation_rate = beta_evaporation_rate
        
        self.distances = None
        self.pheromones = None
        self.heuristics = None
        self.probabilities = None
        
        self.map_coordinates = None
        self.num_nodes = None
        self.not_visited_nodes = None
        
        self.global_best = None # Mảng các cost tốt nhất từ đầu đến vòng lặp thứ i
        self.best_series = None # Mảng các cost tốt nhất của từng vòng lặp
        self.best_path = None # Nghiệm tốt nhất từ trước đến giờ
        self.best = None # Cost tốt nhất từ trước đến giờ
        
        self.fitted = False
        self.fitting_time = None
        
        self.converged = False
        self.stop_iteration = None
    
    def __str__(self) -> str:
        string_ = "Ant Colony Optimizer"
        string_ += "\n------------------------"
        string_ += "\nDesigned to solve TSP. Optimizes either the minimum or maximum distance traveled."
        string_ += "\n------------------------"
        string_ += f"\nNumber of ants:\t\t\t\t{self.ants}"
        string_ += f"\nEvaporation rate:\t\t\t{self.evaporation_rate}"
        string_ += f"\nIntensification factor:\t\t\t{self.Q}"
        string_ += f"\nAlpha Heuristic:\t\t\t{self.heuristic_alpha}"
        string_ += f"\nBeta Heuristic:\t\t\t\t{self.heuristic_beta}"
        string_ += f"\nBeta Evaporation Rate:\t\t\t{self.beta_evaporation_rate}"
        string_ += "\n------------------------"
        
        if self.fitted:
            string_ += '\n\nThis optimizer has been fitted'
        else:
            string_ += '\n\nThis optimizer has not been fitted'
        
        return string_
    
    def __init_information(self, map_coordinates: np.ndarray) -> None:
        for i in map_coordinates:
            assert len(i) == 2, "Have some not valid points"
        
        self.map_coordinates = map_coordinates
        self.num_nodes = len(self.map_coordinates)
        
        self.distances = lib.pairwise_distances(self.map_coordinates, self.map_coordinates)
        
        self.pheromones = np.ones((self.num_nodes, self.num_nodes), dtype=np.float64)
        
        self.heuristics = lib.heuristic_matrix(self.distances)
        
        np.fill_diagonal(self.pheromones, 0)
        np.fill_diagonal(self.heuristics, 0)

        self.probabilities = lib.probability_matrix(self.pheromones, self.heuristics,
                                                    self.heuristic_alpha, self.heuristic_beta)
        
        self.not_visited_nodes = np.arange(0, self.num_nodes)
    
    def __reset_nodes(self) -> None:
        """
        Reset all nodes for next iteration.
        """
        self.not_visited_nodes = np.arange(0, self.num_nodes)
        
    def __update_probabilities(self) -> None:
        """
        Update probabilities matrix each iteration.
        """
        
        self.probabilities = lib.probability_matrix(self.pheromones, self.heuristics,
                                                    self.heuristic_alpha, self.heuristic_beta)
    
       
    def __travel_next_node_from(self, current_node: int) -> int:
        """
        Travel to next node based on probabilities.
        """
        if not isinstance(self.probabilities, np.ndarray):
            self.probabilities = np.array(self.probabilities)
        if not isinstance(self.not_visited_nodes, np.ndarray):
            self.list_of_nodes = np.array(self.list_of_nodes)
        
        allowed_probs = self.probabilities[current_node, self.not_visited_nodes]
        
        # Normalize probabilities
        probabilities = allowed_probs / np.sum(allowed_probs)
        
        next_node = np.random.choice(self.not_visited_nodes, p=probabilities)
        
        return next_node
    
    def __remove_node(self, node: int) -> None:
        """
        Remove node after an ant has visited it.
        """
        
        index = np.where(self.not_visited_nodes == node)[0][0]
        self.not_visited_nodes = np.delete(self.not_visited_nodes, index)
    
    def __evaluate_solution(self, iteration: int, paths: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Evaluate the ant's solution of the current iteration.
        """
        paths_current = paths[iteration]
        
        scores = np.zeros(self.ants, dtype=float)
    
        for i, path in enumerate(paths_current):
            src_nodes = path[:-1]
            destination_nodes = path[1:]
            distances = self.distances[src_nodes, destination_nodes]
            scores[i] = np.sum(distances)
        
        best_index = np.argmin(scores)
        best_path = paths_current[best_index]
        best_score = scores[best_index]
        
        return best_path, best_score
    
    def __evaporate_pheromone(self) -> None:
        self.pheromones = self.pheromones * (1 - self.evaporation_rate)
        
        self.heuristic_beta = self.heuristic_beta * (1 - self.beta_evaporation_rate)
    
    def __intensity(self, best_path: np.ndarray, best_cost: float) -> None:
        
        delta_pheromone = self.Q / best_cost
        
        from_nodes = best_path[:-1]
        to_nodes = best_path[1:]
        
        self.pheromones[from_nodes, to_nodes] += delta_pheromone
    
    def __update_pheromone(self, iteration: int, paths: np.ndarray) -> None:
        """
        Update pheromone on path after ants pass
        """
        paths_current = paths[iteration]
        # L = np.zeros(paths_current.shape[0], dtype=float)
        L_k = 0
        for k, path in enumerate(paths_current):
            from_nodes = path[:-1]
            to_nodes = path[1:]
            distances = self.distances[from_nodes, to_nodes]
            L_k = np.sum(distances)
            
            delta_pheromone = self.Q / L_k
            self.pheromones[from_nodes, to_nodes] += delta_pheromone
    
    def fit(self, map_coordinates: np.ndarray, iterations: int, conv_crit=20, verbose=True) -> None:
        """
        Fit the ACO model to the given map coordinates.

        :param map_coordinates: Coordinates of the cities.
        :param max_iter: Maximum number of iterations.
        """
        
        start = time.time()
        self.__init_information(map_coordinates=map_coordinates)
        
        nums_not_change = 0
        
        if verbose:
            print(f"{self.num_nodes} nodes were given. Beginning ACO Optimization with {iterations} iterations...\n")
            
        self.best_series = np.zeros(iterations, dtype=float)
        self.global_best = np.zeros(iterations, dtype=float)
        
        paths = np.full((iterations, self.ants, self.num_nodes + 1), -1, dtype=int)
        
        path = np.full(self.num_nodes + 1, -1, dtype=int)
        for iteration in range(iterations):
            start_iteration = time.time()
            for ant in range(self.ants):
                current_node = self.not_visited_nodes[np.random.randint(0, self.num_nodes - 1)]
                start_node = current_node
                node_index = 0
                while True:
                    path[node_index] = current_node
                    
                    self.__remove_node(current_node)
                    
                    if self.not_visited_nodes.shape[0] != 0:
                        current_node = self.__travel_next_node_from(current_node=current_node)
                        node_index += 1
                    else:
                        break
                
                path[node_index + 1] = start_node # Add first node to complete HC
                self.__reset_nodes()
                paths[iteration, ant] = path
            
            best_path, best_score = self.__evaluate_solution(iteration=iteration, paths=paths)
            best_path, best_score, improve = three_opt(best_HC=best_path, best_cost=best_score, distances=self.distances)
            # best_path, best_score, improve = local_optimize(best_HC=best_path, best_cost=best_score, distances=self.distances)
            
            if improve:
                self.__intensity(best_path, best_score)
                
            if iteration == 0:
                best_score_so_far = best_score
                self.best_path = best_path
            else:
                if best_score < best_score_so_far:
                    best_score_so_far = best_score
                    self.best_path = best_path
                    
            
            if best_score == best_score_so_far:
                nums_not_change += 1
            else:
                nums_not_change = 0
            # 50 50 45 40 40 30 
            self.best_series[iteration] = best_score
            self.global_best[iteration] = best_score_so_far
            self.__evaporate_pheromone()
            self.__update_pheromone(iteration=iteration, paths=paths)
            self.__update_probabilities()
            
            if verbose:
                print(
                f"Iteration {iteration}/{iterations} | Score: {round(best_score, 3)} | Best so far: {round(best_score_so_far, 3)} | {round(time.time() - start_iteration, 3)} s")
            if (best_score == best_score_so_far) and (nums_not_change == conv_crit):
                self.converged = True
                self.stop_iteration = iteration
                if verbose:
                    print('\nConvergence criterion has been met. Stop')
                break
        
        end = time.time()
        
        if not self.converged:
            self.stop_iteration = iterations
        self.fitting_time = round(end - start)
        self.fitted = True
        
        self.best = self.global_best[-1]
        
        if verbose:
            print(f'\nACO fitted.\tRuntime: {self.fitting_time} seconds.\tBest score: {round(self.best, 3)}')
    
    def get_result(self) -> Tuple[np.ndarray, float, float, np.ndarray]:
        return self.best_path, self.best, self.fitting_time, self.global_best
    
    def plot_best_path(self):
        coordinates = self.map_coordinates[self.best_path]
        plt.plot(coordinates[:, 0], coordinates[:, 1], 'o-')
        plt.title(f"Best Path (Cost: {self.best:.3f})")
        plt.show()
    
    def plot_convergence(self):
        plt.plot(self.global_best, label='Global Best Cost')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Convergence Plot')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    towns = TOWNS
    ACO_optimizer = ACO(ants=ANTS, evaporation_rate=EVAPORATION_RATE, intensification=INTENSIFICATION, alpha=1.0, beta=5.0, beta_evaporation_rate=0.001)
    ACO_optimizer.fit(towns, iterations=MAX_ITER_ACO, conv_crit=20)
    best_path, best, fitting_time, global_best_series = ACO_optimizer.get_result()
    
    best_path_coordinate = [tuple(towns[i]) for i in best_path]
    print(f'Best path: {best_path_coordinate}')
    print(f'Best series: {global_best_series}')
    
    ACO_optimizer.plot_best_path()
    ACO_optimizer.plot_convergence()