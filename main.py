import csv
import re
from clusterize import cluster_file_data
from split_exp import split_experience_column_inplace


def main():
    input_file = 'input.csv'
    buffer_file = 'buffer.csv'

    unique_values = set()
    try:
        filter_unique_routes(input_file, buffer_file)
        print(f"Файл {input_file} успешно обработан. Уникальные маршруты сохранены в {buffer_file}.")
    except FileNotFoundError:
        print(f"Файл {input_file} не найден. Переименуйте выгрузку в 'input.csv'.")
        return
    except Exception as e:
        print(f"Произошла ошибка при обработке файла {input_file}: {e}")
        return
        
    try:    
        clean_experience_column(buffer_file)
    except FileNotFoundError:
        print(f"Файл {buffer_file} не найден.")
    except Exception as e:
        print(f"Произошла ошибка при обработке файла {input_file}: {e}")
        return

    try:    
        filter_by_age_range(buffer_file)
    except FileNotFoundError:
        print(f"Файл {buffer_file} не найден.")
    except Exception as e:
        print(f"Произошла ошибка при обработке файла {input_file}: {e}")
        return


    try:
        split_experience_column_inplace(buffer_file)
    except Exception as e:
        print(f"Произошла ошибка при обработке файла: {e}")


    # Кластеризация DBSCAN
    try:
        cluster_file_data()
    except Exception as e:
        print(f"Произошла ошибка при обработке файла: {e}")

    
    

def filter_unique_routes(i_filename="input.csv", o_filename="buffer.csv"):
    """
    Оставляет только уникальные маршруты по столбцу 'Код маршрута' и сохраняет в новый файл.
    """
    unique_routes = {}
    with open(i_filename, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter=';')
        for row in reader:
            route_code = row['Код маршрута']
            # Сохраняем только первую встреченную запись для каждого маршрута
            if route_code not in unique_routes:
                unique_routes[route_code] = row

    # Записываем результат
    with open(o_filename, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames, delimiter=';')
        writer.writeheader()
        for row in unique_routes.values():
            writer.writerow(row)

def filter_by_age_range(filename='buffer.csv', age_min=30, age_max=40):
    """
    Оставляет только строки, в которых возраст кандидата находится в диапазоне от age_min до age_max.
    Перезаписывает исходный файл.
    """
    with open(filename, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter=';')
        fieldnames = reader.fieldnames
        rows_to_keep = []

        for row in reader:
            age_str = row['Возраст кандидата'].strip()
            if age_str.isdigit():
                age = int(age_str)
                if age_min <= age <= age_max:
                    rows_to_keep.append(row)

    with open(filename, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(rows_to_keep)

    print(f"Оставлены только кандидаты в возрасте от {age_min} до {age_max}.")


def clean_experience_column(filename='buffer.csv', cleaned_filename=None):
    """
    Удаляет строки, в которых в столбце 'Опыт работы кандидата' указано
    'Не отмечено' или 'Опыт работы не указан'. Если cleaned_filename не указан,
    исходный файл будет перезаписан.
    """
    if cleaned_filename is None:
        cleaned_filename = filename  # Перезапись исходного файла

    with open(filename, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter=';')
        fieldnames = reader.fieldnames
        rows_to_keep = []

        for row in reader:
            experience = row['Опыт работы кандидата'].strip().lower()
            if experience not in ['не отмечено', 'опыт работы не указан']:
                rows_to_keep.append(row)

    with open(cleaned_filename, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(rows_to_keep)

    print(f"Очищенные данные от значений 'не отмечено', 'опыт работы не указан' сохранены в '{cleaned_filename}'.")


def contains_letters(text):
    # Проверка наличия хотя бы одной буквы латинского или русского алфавита
    return bool(re.search(r'[A-Za-zА-Яа-яЁё]', text))

if __name__ == '__main__':
    main()