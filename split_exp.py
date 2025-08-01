import csv
import re


def split_experience_column_inplace(input_file='buffer.csv'):
    """
    Разделяет столбец 'Опыт работы кандидата' на 'Компании кандидата' и 'Должности кандидата',
    и перезаписывает исходный файл с добавленными столбцами.
    """
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter=';')
        fieldnames = reader.fieldnames.copy()
        
        # Добавим новые столбцы, если их ещё нет
        if 'Компании кандидата' not in fieldnames:
            fieldnames.append('Компании кандидата')
        if 'Должности кандидата' not in fieldnames:
            fieldnames.append('Должности кандидата')

        updated_rows = []

        for row in reader:
            experience = row.get('Опыт работы кандидата', '').strip().replace('""', '"')
            companies = []
            positions = []

            # Разбиваем по запятой каждого кандидата
            job_parts = re.split(r',(?![^"]*")', experience)

            for part in job_parts:
                # Разделяем по первому дефису
                split_part = re.split(r'\s+-\s+', part.strip(), maxsplit=1)
                if len(split_part) == 2:
                    company, position = split_part
                    companies.append(company.strip())
                    positions.append(position.strip())
                else:
                    pass

            row['Компании кандидата'] = ', '.join(companies)
            row['Должности кандидата'] = ', '.join(positions)
            updated_rows.append(row)

    # Перезапись файла с обновлёнными данными
    with open(input_file, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Файл '{input_file}' обновлён: добавлены столбцы 'Компании кандидата' и 'Должности кандидата'.")

