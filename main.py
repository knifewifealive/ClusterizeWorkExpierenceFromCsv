import csv
import re
from clusterization import clusterize

def contains_letters(text):
    # Проверка наличия хотя бы одной буквы латинского или русского алфавита
    return bool(re.search(r'[A-Za-zА-Яа-яЁё]', text))

def main():
    input_file = 'input.csv'
    output_file = 'buffer.csv'

    unique_values = set()

    
    with open(input_file, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Пропустить заголовок
        for row in csv_reader:
            for cell in row:
                # Разбиваем ячейку по запятой
                for part in cell.split(','):
                    cleaned = re.sub(r'^[^A-Za-zА-Яа-яЁё]+', '', part.strip())
                    if cleaned and contains_letters(cleaned):
                        try:
                            # Оставляем только часть после первого дефиса
                            position = cleaned.split('-', 1)[1].strip()
                            if position:
                                unique_values.add(position)
                        except IndexError:
                            # Если дефиса нет — пропускаем строку
                            continue

    unique_values = sorted(unique_values)

    with open(output_file, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Опыт работы'])  # Заголовок столбца
        for value in unique_values:
            writer.writerow([value])

    clusterize()

if __name__ == '__main__':
    main()