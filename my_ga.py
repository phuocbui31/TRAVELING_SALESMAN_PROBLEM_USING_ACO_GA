import time
import random
import numpy as np
from typing import Tuple
from multiprocessing import Pool, Manager
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
# from parameter_ga import TOWNS

GA_TOURNAMENT_SIZE = 5
GA_MUTATION_RATE = 0.2
GA_MUTATION_DECAY = 1.0
GA_DIVERSITY_THRESHOLD = 0.4
GA_IMPROVEMENT_THRESHOLD = 0.01

class GeneticAlgorithmTSP:
    """Thuật toán di truyền cho bài toán TSP."""

    def __init__(self, population_size: int = 100, tournament_size: int = GA_TOURNAMENT_SIZE,
                 crossover_rate: float = 0.7, mutation_rate: float = GA_MUTATION_RATE,
                 mutation_rate_min: float = 0.015, mutation_decay: float = GA_MUTATION_DECAY, 
                 diversity_threshold: float = GA_DIVERSITY_THRESHOLD,
                 improvement_threshold: float = GA_IMPROVEMENT_THRESHOLD) -> None:
        """Khởi tạo các tham số thuật toán."""
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.improvement_threshold = improvement_threshold

        self.distance_matrix = None
        self.spatial_map = None
        self.num_nodes = None
        self.population = None
        self.distance_cache = None
        self.best_path = None
        self.best_distance = None
        self.best_series = None
        self.global_best = None
        self.fit_time = None
        self.fitted = False
        self.converged = False
        self.stopped_at_iteration = None
        self.current_population_size = population_size

    def calculate_distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """Tính ma trận khoảng cách Euclidean bằng NumPy."""
        distances = cdist(points, points, 'euclidean')
        return np.round(distances)

    def create_random_tour(self) -> np.ndarray:
        """Tạo tour ngẫu nhiên hợp lệ."""
        return np.random.permutation(self.num_nodes).astype(np.int32)

    def create_closest_tour(self, start: int) -> np.ndarray:
        """Tạo tour nearest neighbor từ điểm bắt đầu."""
        tour = [start]
        unvisited = np.ones(self.num_nodes, dtype=bool)
        unvisited[start] = False
        current = start
        for _ in range(self.num_nodes - 1):
            distances = self.distance_matrix[current] * unvisited
            distances[~unvisited] = np.inf
            next_city = np.argmin(distances)
            tour.append(next_city)
            unvisited[next_city] = False
            current = next_city
        return np.array(tour, dtype=np.int32)

    def calculate_len_tour(self, tour: np.ndarray) -> float:
        """Tính khoảng cách tour, trả về np.inf nếu không hợp lệ."""
        indices = np.roll(tour, -1)
        return float(self.distance_matrix[tour, indices].sum())

    def evaluate_tour(self, tour):
        """Đánh giá khoảng cách của một tour."""
        tour_tuple = tuple(tour)
        if tour_tuple in self.distance_cache:
            return self.distance_cache[tour_tuple]
        distance = self.calculate_len_tour(tour)
        self.distance_cache[tour_tuple] = distance
        return distance

    def apply_two_opt(self, tour):
        """Áp dụng 2-opt cho một tour."""
        optimized_tour, optimized_distance = self.two_opt_better(tour)
        return optimized_tour, optimized_distance

    def search_improved_tour_two_opt(self, tour: np.ndarray, current_distance: float) -> Tuple[np.ndarray, float, bool]:
        """Tìm và áp dụng cải tiến 2-opt cho tour."""
        best_tour = tour.copy()
        best_distance = current_distance
        improved = False
        n = len(tour)
        
        for i in range(n - 2):
            for j in range(i + 2, n):
                delta = (
                    - self.distance_matrix[best_tour[i], best_tour[i + 1]]
                    - self.distance_matrix[best_tour[j], best_tour[(j + 1) % n]]
                    + self.distance_matrix[best_tour[i], best_tour[j]]
                    + self.distance_matrix[best_tour[i + 1], best_tour[(j + 1) % n]]
                )
                if delta < -0.01:
                    best_tour[i + 1:j + 1] = best_tour[i + 1:j + 1][::-1]
                    best_distance += delta
                    improved = True
        return best_tour, best_distance, improved

    def two_opt(self, tour: np.ndarray) -> Tuple[np.ndarray, float]:
        """Tối ưu tour bằng 2-opt."""
        best_tour = tour.copy()
        best_distance = self.calculate_len_tour(best_tour)
        best_tour, best_distance, improved = self.search_improved_tour_two_opt(best_tour, best_distance)
        if not improved:
            best_distance = self.calculate_len_tour(best_tour)
        return best_tour, best_distance

    def two_opt_better(self, tour):
        """Tối ưu tour bằng 2-opt cải tiến."""
        n = len(tour)
        improved = True
        
        while improved:
            improved = False
            best_delta = 0
            best_i, best_k = -1, -1
            
            for i in range(1, n-1):
                k_max = n - 1
                for k in range(i+1, k_max):
                    t0 = tour[i-1]
                    t1 = tour[i]
                    tk = tour[k]
                    tk1 = tour[k+1]
                    
                    delta = (self.distance_matrix[t0, t1] + self.distance_matrix[tk, tk1] -
                             self.distance_matrix[t0, tk] - self.distance_matrix[t1, tk1])
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_i, best_k = i, k
                        improved = True
            
            if improved:
                tour[best_i:best_k+1] = tour[best_i:best_k+1][::-1]
        distance = self.calculate_len_tour(tour)
        return tour, distance

    def ox_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Lai ghép Order Crossover (OX)."""
        size = len(parent1)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        child = np.full(size, -1, dtype=np.int32)
        child[start:end] = parent1[start:end]
        remaining = [x for x in parent2 if x not in child[start:end]]
        pos = 0
        for i in range(size):
            if i < start or i >= end:
                child[i] = remaining[pos]
                pos += 1
        return child

    def pmx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Lai ghép PMX."""
        size = len(parent1)
        start, end = sorted(np.random.choice(size, 2, replace=False))
        child = np.full(size, -1, dtype=np.int32)
        child[start:end] = parent1[start:end]
        mapping = {parent1[i]: parent2[i] for i in range(start, end)}
        for i in np.concatenate((np.arange(0, start), np.arange(end, size))):
            candidate = parent2[i]
            while candidate in child and candidate in mapping:
                candidate = mapping[candidate]
            child[i] = candidate
        used = set(child[child != -1])
        remaining = [x for x in range(size) if x not in used]
        child[child == -1] = remaining
        return child

    def two_point_mutation(self, tour: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Đột biến hai điểm: hoán đổi hai thành phố ngẫu nhiên."""
        if random.random() >= mutation_rate:
            return tour
        tour = tour.copy()
        mutation_type = random.random()
        if mutation_type < 0.7:  # Hoán đổi hai điểm (70%)
            i, j = np.random.choice(len(tour), 2, replace=False)
            tour[i], tour[j] = tour[j], tour[i]
        else:  # Xáo trộn đoạn (30%)
            start, end = sorted(np.random.choice(len(tour), 2, replace=False))
            segment = tour[start:end + 1]
            np.random.shuffle(segment)
            tour[start:end + 1] = segment
        return tour

    def tournament_selection(self, fitnesses: np.ndarray) -> np.ndarray:
        """Chọn lọc giải đấu."""
        indices = np.random.choice(self.current_population_size, self.tournament_size, replace=False)
        best_idx = indices[np.argmax(fitnesses[indices])]
        return self.population[best_idx]

    def wheel_selection(self, fitnesses: np.ndarray) -> np.ndarray:
        """Chọn lọc bánh xe độ thích nghi."""
        scaled_fitnesses = np.maximum(0, fitnesses - fitnesses.min())
        total_fitness = scaled_fitnesses.sum()
        if total_fitness == 0:
            return self.population[np.random.choice(self.current_population_size)]
        probabilities = scaled_fitnesses / total_fitness
        selected_idx = np.random.choice(self.current_population_size, p=probabilities)
        return self.population[selected_idx]

    def init_population(self, spatial_map: np.ndarray, manager: Manager) -> None:
        """Khởi tạo quần thể và ma trận khoảng cách."""
        self.spatial_map = spatial_map
        assert self.spatial_map.shape[1] == 2, "Tọa độ không hợp lệ! Mỗi điểm phải có giá trị x, y."
        self.num_nodes = len(self.spatial_map)
        self.distance_matrix = self.calculate_distance_matrix(self.spatial_map)
        self.population = []
        nn_count = int(self.population_size * 0.1)

        # Song song hóa tạo tour nearest neighbor
        with Pool() as pool:
            nn_tours = pool.map(self.create_closest_tour, [random.randint(0, self.num_nodes-1) for _ in range(nn_count)])
        self.population.extend(nn_tours)

        # Tạo tour ngẫu nhiên
        for _ in range(self.population_size - nn_count):
            tour = self.create_random_tour()
            self.population.append(tour)
        self.population = np.array(self.population, dtype=np.int32)
        self.current_population_size = len(self.population)

        # Khởi tạo distance_cache với Manager.dict
        self.distance_cache = manager.dict()
        with Pool() as pool:
            distances = pool.map(self.evaluate_tour, self.population)
        for tour, distance in zip(self.population, distances):
            self.distance_cache[tuple(tour)] = distance

        initial_distances = np.array(list(self.distance_cache.values()))
        print(f"Quần thể ban đầu - Khoảng cách nhỏ nhất: {initial_distances.min():.2f}, "
              f"Khoảng cách trung bình: {initial_distances.mean():.2f}")

    def generate_offspring(self, parameters):
        """Tạo một cá thể con."""
        fitnesses, crossover_type = parameters
        parent1 = self.tournament_selection(fitnesses) if random.random() <= 0.5 else self.wheel_selection(fitnesses)
        parent2 = self.tournament_selection(fitnesses) if random.random() <= 0.5 else self.wheel_selection(fitnesses)
        if random.random() > self.crossover_rate:
            child = parent1.copy()
        else:
            if crossover_type == 'ox':
                child = self.ox_crossover(parent1, parent2)
            else:
                child = self.pmx_crossover(parent1, parent2)
        child = self.two_point_mutation(child, self.mutation_rate)
        distance = self.calculate_len_tour(child)
        return child, distance

    def fit(self, spatial_map: np.ndarray, generations: int, conv_crit: int = 200,
            verbose: bool = True) -> None:
        """Huấn luyện thuật toán di truyền."""
        start = time.time()
        manager = Manager()
        self.init_population(spatial_map, manager)
        num_no_improvement = 0
        previous_best_distance = np.inf
        self.best_series = []
        self.global_best = []
        convergence_threshold = 1e-4
        time_iteration = 0
        
        if verbose:
            print(f"{self.num_nodes} nút được cung cấp. Bắt đầu GA với 2-opt qua {generations} thế hệ...\n")

        with Pool() as pool:
            for generation in range(generations):
                start_iter = time.time()
                self.current_population_size = len(self.population)
                
                if generation != 0:
                    print('Begin applying 2-opt...')
                    # Áp dụng 2-opt song song
                    top_count = int(self.population_size * 0.5)
                    top_indices = np.argsort(distances)[:top_count]
                    population_to_local_search = self.population[top_indices]
                        
                    results = pool.map(self.apply_two_opt, population_to_local_search)
                    for i, (two_opt_path, two_opt_distance) in enumerate(results):
                        self.population[i] = two_opt_path
                        tour_tuple = tuple(two_opt_path)
                        self.distance_cache[tour_tuple] = two_opt_distance
                    print('End applying 2-opt')

                # Tính fitness của các cá thể
                distances = np.array(pool.map(self.evaluate_tour, self.population))
                fitnesses = np.where(distances > 0, 1.0 / distances, np.inf)

                # Lưu cá thể tốt nhất
                current_best_idx = np.argmin(distances)
                current_best_tour = self.population[current_best_idx]
                current_best_distance = distances[current_best_idx]

                if current_best_distance < (self.best_distance or np.inf):
                    improvement = previous_best_distance - current_best_distance
                    self.best_distance = current_best_distance
                    self.best_path = current_best_tour.copy()
                    if improvement > self.improvement_threshold:
                        num_no_improvement = 0
                else:
                    num_no_improvement += 1
                    improvement = 0.0

                previous_best_distance = self.best_distance or np.inf
                self.best_series.append(current_best_distance)
                self.global_best.append(self.best_distance)

                if verbose:
                    print(
                        f"Thế hệ {generation}/{generations} | Khoảng cách: {round(current_best_distance, 3)} | "
                        f"Tốt nhất: {round(self.best_distance, 3)} | Tỷ lệ đột biến: {round(self.mutation_rate, 4)} | "
                        f"Kích thước quần thể: {self.current_population_size} | "
                        f"{round(time_iteration, 3)} s | Không cải thiện: {num_no_improvement}"
                    )
                
                # Kiểm tra điều kiện dừng
                if num_no_improvement >= conv_crit:
                    self.converged = True
                    self.stopped_at_iteration = generation
                    if verbose:
                        print("\nTiêu chí hội tụ đạt được. Dừng lại...")
                    break

                if generation > 50 and np.abs(self.global_best[generation] - self.global_best[generation - 50]) < convergence_threshold:
                    self.converged = True
                    self.stopped_at_iteration = generation
                    if verbose:
                        print("\nKhoảng cách tốt nhất ổn định. Dừng sớm...")
                    break

                # Tạo con cái song song
                crossover_choices = np.random.choice(['ox', 'pmx'], size=self.population_size, p=[0.5, 0.5])
                offspring_results = pool.map(self.generate_offspring, 
                                            [(fitnesses, crossover_choices[i]) for i in range(self.population_size)])
                offspring = np.array([result[0] for result in offspring_results])
                for child, distance in offspring_results:
                    self.distance_cache[tuple(child)] = distance

                # Gộp quần thể cha mẹ và con
                combined_population = np.vstack((self.population, offspring))
                combined_distances = np.array(pool.map(self.evaluate_tour, combined_population))

                # Chọn lọc elitism
                sorted_indices = np.argsort(combined_distances)
                elite_count = int(self.population_size * 0.95)
                random_count = self.population_size - elite_count
                selected_indices = sorted_indices[:elite_count].tolist()
                remaining_indices = sorted_indices[elite_count:].tolist()
                if remaining_indices:
                    selected_indices.extend(random.sample(remaining_indices, min(random_count, len(remaining_indices))))
                self.population = combined_population[selected_indices]
                self.current_population_size = len(self.population)

                end_iter = time.time()
                time_iteration = end_iter - start_iter


        if not self.converged:
            self.stopped_at_iteration = generations
        self.fit_time = round(time.time() - start)
        self.fitted = True
        self.best_path = np.append(self.best_path, self.best_path[0])

        if verbose:
            print(
                f"\nGA với 2-opt đã huấn luyện xong. Thời gian chạy: {self.fit_time // 60} phút. "
                f"Khoảng cách tốt nhất: {round(self.best_distance, 3)}"
            )

    def get_result(self) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray, int, float, float]:
        """Trả về kết quả."""
        return self.best_path, self.best_distance, self.fit_time, self.global_best, self.best_series, self.population_size, self.crossover_rate, self.mutation_rate

    def plot_best_path(self):
        """Vẽ đường đi tốt nhất."""
        coordinates = self.spatial_map[self.best_path]
        plt.plot(coordinates[:, 0], coordinates[:, 1], 'o-')
        plt.title(f"Best Path (Cost: {self.best_distance:.3f})")
        plt.show()
    
    def plot_convergence(self):
        """Vẽ biểu đồ hội tụ."""
        plt.plot(self.global_best, label='Global Best Cost', marker='^')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Convergence Plot')
        plt.legend()
        plt.show()

def run_GA(pop_size: int, TOWNS, crossover_rate: float = 0.8, mutation_rate: float = 0.1):
    """Hàm chính để chạy thuật toán và lưu kết quả."""
    ga = GeneticAlgorithmTSP(population_size=pop_size, crossover_rate=crossover_rate, mutation_rate=mutation_rate)
    ga.fit(spatial_map=TOWNS, generations=1000, conv_crit=100, verbose=True)
    best_path, best_distance, fit_time, global_bests, local_bests, pop_size, crossover_rate, mutation_rate = ga.get_result()
    best_path_coor = [tuple(TOWNS[i]) for i in best_path]
    
    print("Chỉ số đường đi tốt nhất:", best_path)
    print("Tọa độ đường đi tốt nhất:", best_path_coor)
    print("Khoảng cách tốt nhất:", round(best_distance, 2))
    print("Thời gian chạy:", fit_time, "giây")
    print("Global bests:", global_bests)
    print('Population size:', pop_size)
    print('Crossover rate: ', crossover_rate)
    print('Mutation rate: ', mutation_rate)
    # ga.plot_best_path()
    # ga.plot_convergence()
    
    return fit_time, best_distance, global_bests, local_bests, best_path_coor, pop_size, crossover_rate, mutation_rate