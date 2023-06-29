import sklearn as sk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score # для определения качества модели
from sklearn.model_selection import train_test_split

data_set = pd.read_csv("insurance.csv")
print(data_set.head())
print(data_set.describe())

scaler_mnmx = sk.preprocessing.MinMaxScaler()
sex_code = {"female": 0, "male": 1}
region_code = {"southwest": 0, "northwest": 1, "northeast": 2, "southeast": 3}
smoke_code = {"yes": 1, "no": 0}
data_set["sex"] = data_set["sex"].replace(sex_code)
data_set["region"] = data_set["region"].replace(region_code)
data_set["smoker"] = data_set["smoker"].replace(smoke_code)
data_set["age"] = scaler_mnmx.fit_transform(data_set[["age"]])
data_set["bmi"] = scaler_mnmx.fit_transform(data_set[["bmi"]])
data_set["children"] = scaler_mnmx.fit_transform(data_set[["children"]])
data_set["charges"] = scaler_mnmx.fit_transform(data_set[["charges"]])
print(data_set.head())
charges = data_set.loc[:,["charges"]]
for i in range(len(charges)):
    if charges.iloc[i,0] <  0.25:
        charges.iloc[i,0] = 0
    elif charges.iloc[i,0] < 0.5:
        charges.iloc[i,0] = 1
    elif charges.iloc[i,0] < 0.75:
        charges.iloc[i,0] = 2
    else:
        charges.iloc[i,0] = 3
data_set["charges"] = charges

print(data_set.head())

X_train, X_test, y_train, y_test = train_test_split(
    # поскольку наша бд это pandas-таблица, для нее нужно указыать iloc
    data_set.loc[:, "age":"region"],  # берем все колонки кроме последней
    data_set.loc[:, "charges"], # последнюю в целевую переменную(класс)(то что будем предсказывать)
    test_size = 0.20 # Размер тестовой выборки 20%
)


# Обучим метод к ближйшх соседей
model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train, y_train)

# Получим предсказание модели
y_pred = model.predict(X_test)

# Покажем на графике, что скажет полученное число.
# Красным цветом обозначим точки, для которых классификация сработала неправильно.
plt.figure(figsize=(5,3))
sns.scatterplot(x = 'age', y = 'smoker', data=data_set, hue='charges', s = 70)

cm = sk.metrics.confusion_matrix(y_test, y_pred) # Матрица ошибок
ac = sk.metrics.accuracy_score(y_test, y_pred)
print(cm, ac)
print(f'accuracy: {accuracy_score(y_test, y_pred) :.2}')