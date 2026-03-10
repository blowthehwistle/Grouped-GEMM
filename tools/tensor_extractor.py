from pathlib import Path
def process_file(input_filename, output_filename):
    # 입력 파일 읽기
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    current_case = []
    correct_cases = []

    # 올바른 케이스 찾기
    for line in lines:
        line = line.strip()
        if line.startswith('('):  # New case starts
            current_case = [line]
        elif line:  # Non-empty line
            current_case.append(line)
            if line == 'Result is correct':
                correct_cases.append(current_case)  # copy() 사용하여 독립적인 리스트 생성

    # 마지막 수치를 기준으로 내림차순 정렬
    # 각 케이스의 마지막 수치를 float로 변환하여 정렬
    correct_cases.sort(key=lambda x: float(x[-1]), reverse=True)

    # 결과를 output.txt 파일에 쓰기
    Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
    with open(output_filename, 'w') as outfile:
        outfile.write("# of correct results : {}\n\n".format(len(correct_cases)))
        for case in correct_cases:
            for line in case:
                outfile.write(line + '\n')
            outfile.write('\n')  # 각 케이스 사이에 빈 줄 추가

# 파일 사용 예시
input_filename = str(Path(__file__).resolve().parents[1] / "results" / "benchmark" / "tensor_5_double_buffering_4k_t3.txt")  # 입력 파일 이름
output_filename = str(Path(__file__).resolve().parents[1] / "results" / "tuned" / "tensor_5_double_buffering_4k_t3.txt")  # 출력 파일 이름
process_file(input_filename, output_filename)