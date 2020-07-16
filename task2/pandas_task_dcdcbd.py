import pandas as pd

people = pd.read_csv("people.csv")
cars = pd.read_csv("cars.csv")
appartments = pd.read_csv("appartments.csv")


# Task 1(d): Заменить пропущенные значения медианой


def fillna_median(df):
    for i in df:
        if df[i].isnull().sum() > 0:
            df[i + '_indicator'] = pd.isnull(df[i])
    return df.fillna(df.median())


people = fillna_median(people)
cars = fillna_median(cars)
appartments = fillna_median(appartments)

people.to_csv("people_stage1.csv", index=False)
cars.to_csv("cars_stage1.csv", index=False)
appartments.to_csv("appartments_stage1.csv", index=False)

# Task 2(c): Добавить в таблицу people бинарный признак того,
# что у человекае сть хотя бы одна квартира или хотя бы одна машина

people['Has_appartment_or_car'] = 0
has_car = cars['ID_person'].unique().tolist()
has_appartment = appartments['ID_person'].unique().tolist()
has_appartment_or_car = list(set(has_car + has_appartment))
people.loc[has_appartment_or_car, 'Has_appartment_or_car'] = 1
people.to_csv("people_stage2.csv", index=False)

# Task 3(d): Отсортировать таблицу cars по доходу владельца по убыванию

cars['Salary_person'] = people.loc[cars['ID_person'], 'Salary'].values
cars = cars.sort_values('Salary_person', ascending=False)
cars = cars.drop('Salary_person', 1)
cars.to_csv("cars_stage3.csv", header=False)

# Task 4(cb): Посчитать среднее количество стоимости машин для студентов

students_id = []
students_id = people.index[(people['Profession'] == "студент") & (people['Sex'] == 'м')].tolist()
average_price = cars.loc[cars['ID_person'].isin(students_id), 'Price'].sum() / len(students_id)
print(average_price)

# Task 5(d): Сохранить таблиц уминимальных максимальных зарплат в зависимости от пола, профессии и количествамашин

people['Number_of_cars'] = 0
people.loc[cars['ID_person'], 'Number_of_cars'] = cars['ID_person'].value_counts()
grouped = people.groupby(['Sex', 'Profession', 'Number_of_cars'], as_index=False).agg({"Salary": [max, min]})
grouped.columns = ["_".join(x) for x in grouped.columns.ravel()]
grouped.to_csv("stage5.csv")
