def compare_files(file1_path, file2_path):
    """
    Сравнивает строки из первого файла со вторым файлом, учитывая различия в формате.
    Возвращает список отсутствующих строк и проверяет наличие всех строк из первого файла во втором.
    """
    
    # Читаем строки из первого файла
    with open(file1_path, 'r', encoding='utf-8') as f1:
        file1_lines = [line.strip() for line in f1 if line.strip()]
    
    # Читаем строки из второго файла
    with open(file2_path, 'r', encoding='utf-8') as f2:
        file2_lines = [line.strip() for line in f2 if line.strip()]
    
    # Создаем множество для быстрого поиска ключевых частей из второго файла
    file2_keys = set()
    
    for line in file2_lines:
        # Пропускаем строки с PROMPT и пустые строки
        if line.startswith('PROMPT:') or not line:
            continue
            
        # Разделяем строку по табуляции и берем первую часть (путь к файлу)
        parts = line.split('\t')
        if parts:
            file_path = parts[0]
            # Извлекаем ключевую часть после 'test_videos/'
            if 'test_videos/' in file_path:
                key = file_path.split('test_videos/')[1]
                file2_keys.add(key)
    
    # Проверяем наличие каждой строки из первого файла во втором
    missing_lines = []
    
    for line in file1_lines:
        # Извлекаем ключевую часть из строки первого файла
        if 'test_videos/' in line:
            key = line.split('test_videos/')[1]
        else:
            key = line
            
        if key not in file2_keys:
            missing_lines.append(line)
    
    return missing_lines

def main(file2_path: str):
    file1_path = 'XDV_Test.txt'  # путь к первому файлу
    
    missing_lines = compare_files(file1_path, file2_path)
    print("Проверка файла", file2_path)
    if not missing_lines:
        print("✓ Все строки из первого файла присутствуют во втором файле!")
    else:
        print(f"✗ Найдено {len(missing_lines)} отсутствующих строк:")
        for line in missing_lines:
            print(f"  - {line}")
    print("\n")

if __name__ == "__main__":
    files = [
        # 'results/Qwen2.5-VL-3B-Instruct_basic_xd-violence.txt',
        # 'results/Qwen2.5-VL-3B-Instruct_cot_xd-violence.txt',
        # 'results/Qwen2.5-VL-3B-Instruct_fewshot_xd-violence.txt',
        # 'results/Qwen2.5-VL-3B-Instruct_zero_shot_xd-violence.txt',
        'results/gemma-3n-E2B-it_basic_xd-violence.txt',
        'results/gemma-3n-E2B-it_cot_xd-violence.txt',
        'results/gemma-3n-E2B-it_fewshot_xd-violence.txt',
        'results/gemma-3n-E2B-it_zero_shot_xd-violence.txt',

    ]
    for path in files:
        main(path)
