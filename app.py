from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
import numpy as np  # Доданий імпорт numpy

diabetes = datasets.load_diabetes()

print(diabetes.DESCR)

db_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
print(db_df.sample(5))
db_df['Progression'] = diabetes.target  # Нове ім'я стовпця 'Progression'
print(db_df.sample(2))  # Перевірка набору даних ще раз.

# Незалежні змінні / пояснюючі змінні
x = db_df.drop(labels='Progression', axis=1)  # axis=1 означає, що ми видаляємо дані за стовпцем.
# Залежна змінна / відгук / цільова змінна.
y = db_df['Progression']

# Розбиваємо набір даних на 75%-25% тренувально-тестовий розбиток
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=999)
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

# Імпортуємо лінійну регресію з sklearn та створюємо екземпляр моделі.
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

print(lm)
print(type(lm))
lm.fit(train_x, train_y)
predicted_y = lm.predict(test_x)

print("1) The model explains,", np.round(mt.explained_variance_score(test_y, predicted_y) * 100, 2),
      "% variance of the target w.r.t features is")
print("2) The Mean Absolute Error of model is:", np.round(mt.mean_absolute_error(test_y, predicted_y), 2))
print("3) The R-Square score of the model is ", np.round(mt.r2_score(test_y, predicted_y), 2))
