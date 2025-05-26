import argparse
import json
import numpy as np
import os
from parameter_ga import read_map
from my_ga import run_GA

def convert_to_serializable(obj):
    """Chuyển đổi các đối tượng không serializable thành dạng JSON-compatible."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj

def run_and_save(TOWNS, n, i, output, filename, crossover_rate_input=0.9):
    print(f'\n===================================== Running test {i + 1} =====================================')
    fit_time, best_distance, global_bests, local_bests, best_path_coor, pop_size, crossover_rate, mutation_rate = run_GA(
        pop_size=n, TOWNS=TOWNS, crossover_rate=crossover_rate_input, mutation_rate=0.3)
    
    output[f'Test {i+1}'] = {
            'fit_time': fit_time,
            'best_distance': best_distance,
            'global_bests': global_bests,
            'local_bests': local_bests,
            'best_path_coor': best_path_coor,
            'pop_size': pop_size,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate
    }
        
    serializable_output = {}
    for test, data in output.items():
        serializable_output[test] = {
            key: convert_to_serializable(value)
            for key, value in data.items()
        }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_output, f, indent=4)
    print(f"Đã lưu output vào {filename}")


if __name__ == "__main__":
    # Thiết lập parser để nhận đối số dòng lệnh
    parser = argparse.ArgumentParser(description="Run GA for TSP with data file path")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the TSP data file')
    args = parser.parse_args()

    # Đọc dữ liệu từ file .tsp được truyền qua đối số
    TOWNS = read_map(args.data_path)
    print(args.data_path)
    output = {}
    n = TOWNS.shape[0]
    
    output = {}
    output_dir = "results"
    clean_path = os.path.basename(args.data_path).replace('/', '').replace('\\', '').replace('Benchmark', '')
    clean_path = clean_path.replace('.tsp', '')
    filename = os.path.join(output_dir, f"output_{clean_path}_.json")

    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(15):
        if i < 5:
            run_and_save(TOWNS, int(n/2), i, output, filename, crossover_rate_input=0.9)
        elif i < 10:
            run_and_save(TOWNS, n, i, output, filename, crossover_rate_input=0.9)
        else:
            run_and_save(TOWNS, int(n*1.5), i, output, filename, crossover_rate_input=0.9)