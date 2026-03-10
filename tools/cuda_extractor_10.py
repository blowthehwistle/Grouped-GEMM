from pathlib import Path
import re

def parse_data(file_path):
    results = []
    current_data = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'\((\d+)/(\d+)\)', line)
            if match:
                if current_data and 'is_correct' in current_data and current_data['is_correct']:
                    results.append(current_data)
                current_data = {'step': int(match.group(1)), 'total_steps': int(match.group(2))}
            
            if 'Result is correct' in line:
                current_data['is_correct'] = True
            elif 'Kernel Execution Time' in line:
                current_data['kernel_time'] = float(line.split(':')[1].split()[0])
            elif 'Kernel Performance' in line:
                current_data['kernel_gflops'] = float(line.split(':')[1].split()[0])
            elif 'cuBLAS Execution Time' in line:
                current_data['cublas_time'] = float(line.split(':')[1].split()[0])
            elif 'cuBLAS Performance' in line:
                current_data['cublas_gflops'] = float(line.split(':')[1].split()[0])
            elif 'BK=' in line:
                params = line.split()
                for param in params:
                    if '=' in param:  # Only process if it contains an '='
                        key, value = param.split('=')
                        current_data[key.lower()] = int(value)
    
    if current_data and 'is_correct' in current_data and current_data['is_correct']:
        results.append(current_data)
    
    return results

def save_results(results, output_file):
    # Sort results by GFLOPS ratio (Kernel GFLOPS / cuBLAS GFLOPS) in descending order
    sorted_results = sorted(results, key=lambda x: x['kernel_gflops'] / x['cublas_gflops'], reverse=True)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as file:
        file.write(f"Total correct results: {len(sorted_results)}\n\n")
        file.write("="*80 + "\n")
        for data in sorted_results:
            relative_percentage = (data['kernel_gflops'] / data['cublas_gflops'] * 100)
            file.write(f"Step: {data['step']}/{data['total_steps']}\n")
            file.write(f"cuBLAS Execution Time : {data['cublas_time']:.7f} seconds\n")
            file.write(f"cuBLAS Performance : {data['cublas_gflops']:.2f} GFLOPS\n")
            file.write(f"Kernel Execution Time : {data['kernel_time']:.9f} seconds\n")
            file.write(f"Kernel Performance : {data['kernel_gflops']:.2f} GFLOPS\n")
            file.write(f"Relative Percentage : {relative_percentage:.6f}%\n")
            file.write(f"BM : {data.get('bm', 'N/A')}\n")
            file.write(f"BN : {data.get('bn', 'N/A')}\n")
            file.write(f"BK : {data.get('bk', 'N/A')}\n")
            file.write(f"WM : {data.get('wm', 'N/A')}\n")
            file.write(f"WN : {data.get('wn', 'N/A')}\n")
            file.write(f"WNITER : {data.get('wn_iter', 'N/A')}\n")
            file.write(f"TW_M : {data.get('tm', 'N/A')}\n")
            file.write(f"TW_N : {data.get('tn', 'N/A')}\n")
            file.write(f"NUM_THREADS : {data.get('num_threads', 'N/A')}\n")
            file.write("="*80 + "\n")

# Example usage:
input_file = str(Path(__file__).resolve().parents[1] / 'results' / 'benchmark' / '11_warptiling_4k_t1.txt')
output_file = str(Path(__file__).resolve().parents[1] / 'results' / 'tuned' / '10_warptiling_4k_t1.txt')  # Path to save the filtered output

parsed_data = parse_data(input_file)
save_results(parsed_data, output_file)
