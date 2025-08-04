import pandas as pd

df = pd.read_csv('input.csv', delimiter=';')
df = df.dropna(subset=['Опыт работы кандидата','Возраст кандидата', 'Город кандидата'])
df = df.drop_duplicates(subset=['Код маршрута'])
df['Возраст кандидата'] = pd.to_numeric(df['Возраст кандидата'], errors='coerce')

keyword_weights = {
    'Розничные продажи': {
        'keywords': [
            'ведущий менеджер по продажам', 'эксперт по продажам', 'специалист по клиентским продажам',
            'консультант по премиум-продажам', 'лидер розничных продаж', 'стратег продаж',
            'специалист по работе с vip-клиентами', 'менеджер по развитию клиентской базы',
            'эксперт по увеличению продаж', 'координатор розничных сделок', 'специалист по продажам люксовых товаров', 'b2c продаж', 'менеджер по работе с клиентами',
            'менеджер по продажам', 'продаж'
        ],
        'weight': 10.00
    },
    'Индивидуальный предприниматель': {
        'keywords': [
            'ип', 'индивидуальный предприниматель', 'фриланс', 'самозанятый',
            'предприниматель', 'собственный бизнес', 'стартап', 'малый бизнес'
        ],
        'weight': 5.17
    },
    'Недвижимость': {
        'keywords': [
            'недвижимость', 'риелтор', 'риэлтор', 'агент по недвижимости', 'брокер',
            'продажа недвижимости', 'аренда недвижимости', 'оценка недвижимости',
            'ипотека', 'жилые помещения', 'коммерческая недвижимость'
        ],
        'weight': 4.55
    },
    'Банковские, кредитные и финансовые специалисты': {
        'keywords': [
            'банк', 'банковский', 'кредит', 'финансовый', 'бухгалтер',
            'кредитный специалист', 'финансовый аналитик', 'экономист',
            'инвестиции', 'страхование', 'аудит', 'кредитный менеджер'
        ],
        'weight': 3.75
    },
    'Недвижимость (из другого агентства недвижимости)': {
        'keywords': [
            'агентство недвижимости', 'специалист по недвижимости', 'риелторская компания',
            'продажа жилья', 'аренда жилья', 'недвижимость премиум', 'элитная недвижимость',
            'управление недвижимостью', 'девелопмент', 'застройщик'
        ],
        'weight': 3.62
    },
    'Административный персонал': {
        'keywords': [
            'администратор', 'офис-менеджер', 'секретарь', 'ассистент',
            'координатор', 'ресепшионист', 'управление офисом', 'административная поддержка'
        ],
        'weight': 2.93
    },
    'Услуги для населения': {
        'keywords': [
            'услуги', 'работа с клиентами', 'обслуживание клиентов', 'консультации',
            'сервис', 'клиентская поддержка', 'парикмахер', 'мастер маникюра',
            'фитнес-тренер', 'гид', 'аниматор'
        ],
        'weight': 2.48
    },
    'Оптовые продажи': {
        'keywords': [
            'оптовые продажи', 'опт', 'торговый агент', 'менеджер по оптовым продажам',
            'закупки', 'поставки', 'дистрибьютор', 'логистика продаж',
            'складская логистика', 'b2b продажи', 'коммерческий представитель'
        ],
        'weight': 2.31
    }
}

def calculate_score_and_keywords(experience, age) -> tuple:
    if pd.isna(experience) or pd.isna(age):
        return 0, ''
    
    # Calculate experience-based score
    experience = str(experience).lower()
    score = 0
    matched_keywords = []
    for category, data in keyword_weights.items():
        for keyword in data['keywords']:
            if keyword in experience and keyword not in matched_keywords:
                score += data['weight']
                matched_keywords.append(keyword)
    
    # Apply age-based multiplier
    if age < 18:
        score = 0  # Candidates under 18 are not considered
    elif 30 <= age <= 40:
        score *= 1.0  # Full weight for 30-40 age range
    elif 18 <= age < 30:
        score *= 0.7  # Reduced weight for younger candidates
    else:  # age > 40
        score *= 0.85  # Moderately reduced weight for older candidates
    
    return score, ', '.join(matched_keywords)

df['Опыт работы кандидата'] = df['Опыт работы кандидата'].str.lower()

# Apply scoring function with both experience and age
df[['Score', 'keywords']] = df.apply(
    lambda row: calculate_score_and_keywords(row['Опыт работы кандидата'], row['Возраст кандидата']),
    axis=1
).apply(pd.Series)

# Filter candidates with non-zero scores and sort by score
df = df[df['Score'] > 0].sort_values(by='Score', ascending=False)
df.to_csv('filter_candidates.csv', sep=';', index=False)