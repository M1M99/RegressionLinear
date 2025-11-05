import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_excel('youtube_data.xlsx')
print(df.head())

describe_data = df.describe()
print(describe_data)
corr_matrix = df[['Likes', 'Comments', 'Views']].corr()
print("CORR",corr_matrix)
#en cox tesir eden likes
X = df["Likes"].values
y = df["Views"].values
#likes ile view arasinda guclu elaqe var
#views ile comment arsinda orta elaqe var
#region Scatter
plt.scatter(X, y)
plt.xlabel("Likes")
plt.ylabel("Views")
plt.title('Likes and Views Correlation')
plt.show()
#endregion
m = np.cov(X, y, bias=True)[0, 1] / np.var(X)
b = np.mean(y) - m * np.mean(X)

print("m:", m)
print("b: intercept", b)
pred= m*X+b
print(pred)
mae =  mean_absolute_error(y, pred)
mse =mean_squared_error(y, pred)
r2 = r2_score(y, pred)
print("MAE:", mae, "MSE:", mse, "R2:", r2)
X = df[["Likes", "Comments"]]
y = df["Views"]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print('R2',r2)
print("Coef:", model.coef_)
print("Intercept:", model.intercept_)


X = df['Likes'].values
Y = df['Views'].values

m = 0
b = 0
# L = 0.0001 overflow
L = 0.00000001 #normal rate
epochs = 1000
n = len(X)

errors = []

for i in range(epochs):
    Y_pred = m * X + b
    error = np.mean((Y - Y_pred) ** 2)
    errors.append(error)

    D_m = (-2 / n) * sum(X * (Y - Y_pred))
    D_b = (-2 / n) * sum(Y - Y_pred)
    m = m - L * D_m
    b = b - L * D_b

    if i % 100 == 0:
        print(f"Epoch {i}: Error={error:.2f}")

print(f"\nFinal m={m}, b={b}")

plt.plot(errors,color='darkblue')
plt.title('Error vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

#7. texmin et

likes_count = int(input('Likes Count'))
comments_count = int(input('Comments Count'))

new_df = pd.DataFrame({
    'Likes': [likes_count],
    'Comments': [comments_count]
})

pred= model.predict(new_df)[0]
print('Texmini View',pred)

X_values = df[['Likes', 'Comments']]
y_pred_all = model.predict(X_values)
r2 = r2_score(y, y_pred_all)
print('R2 Score:', r2)
#multi linear daha yuksek dogrulugla isleyi(linearla muqayisede)
#r2 score 99% ideal isleyir