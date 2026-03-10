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
                    key, value = param.split('=')
                    current_data[key.lower()] = int(value)
    
    if current_data and 'is_correct' in current_data and current_data['is_correct']:
        results.append(current_data)
    
    return results

def save_results(results, output_file):
    # 결과를 GFLOPS 비율에 따라 내림차순으로 정렬
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
            file.write(f"BM : {data['bm']}\n")
            file.write(f"BN : {data['bn']}\n")
            file.write(f"BK : {data['bk']}\n")
            file.write(f"TW_M : {data['tm']}\n")
            file.write(f"TW_N : {data['tn']}\n")
            file.write(f"NUM_THREADS : {data['num_threads']}\n")
            file.write("="*80 + "\n")

# 사용 예시
kernels = ['6_vectorized.txt']
#kernels = ['10_warptiling.txt']
input_file = str((Path(__file__).resolve().parents[1] / 'results' / 'benchmark')) + '/'  # 입력 파일 경로
output_file = str((Path(__file__).resolve().parents[1] / 'results' / 'tuned')) + '/'  # 출력 파일 경로

for kernel in kernels:
    input_path = input_file + kernel
    output_path = output_file + kernel
    parsed_data = parse_data(input_path)
    save_results(parsed_data, output_path)