import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# area=np.array([50,60,70,80,90,100])
# price = np.array([150000,180000,210000,220000,230000,240000])
# #
# # a = np.cov(area,price,bias=True)[0,1]/np.var(area)
# # b = np.mean(price)-a*np.mean(area)

# # print(a)
# #
# # print(b)
# #
# # pred= a*area+b
# # print(pred)
# # # plt.scatter(area,price,color='teal',label='Real Data')
# # # plt.plot(area,price,color='red',label='Regression Liner')
# # # plt.legend()
# # # plt.title('Real Data')
# # # plt.xlabel('Area')
# # # plt.ylabel('Price')
# # # plt.show()
# #
# from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
#
# # # mae = mean_absolute_error(price,pred)
# # # print(mae)
# # #
# # # r2 = r2_score(price,pred)
# # # print(r2)
# #
# #
# from fontTools.misc.timeTools import epoch_diff
#
# X=area
# Y=price
#
# m=0
# b=0
# L=0.0001
# epochs = 3000
# n=len(X)
# for i in range(epochs):
#     Y_pred = m*X + b
#     D_m=(-2/n)*sum(X*(Y-Y_pred))
#     D_b = (-2/n)*sum(Y-Y_pred)
#     m=m-L*D_m
#     b=b-L*D_b
#
# print(m)
# print(b)
#
# mae = mean_absolute_error(Y,Y_pred)
# mse =mean_squared_error(Y,Y_pred)
# print(mae)
#
# r2 = r2_score(Y,Y_pred)
# print(r2)

df = pd.read_excel('youtube_data.xlsx')
print(df.head())

describe_data = df.describe()
print('CORR',df[['Likes','Comments','Views']].corr())
print(describe_data)
corr_matrix = df[['Likes', 'Comments', 'Views']].corr()

print(corr_matrix)
#likes
X = df["Likes"].values
y = df["Views"].values

m = np.cov(X, y, bias=True)[0, 1] / np.var(X)
b = np.mean(y) - m * np.mean(X)

print("m:", m)
print("b:", b)
pred= m*X+b
print(pred)
X = df[["Likes", "Comments"]]
y = df["Views"]

model = LinearRegression()
model.fit(X, y)

print("Coef:", model.coef_)
print("Intercept:", model.intercept_)
# a = np.cov(X,y,bias=True)[0,1]/np.var(X)
# b = np.mean(y)-a*np.mean(X)
# print(a)
# print(b)
# linear_regression.
# print("COef:" ,linear_regression.coef_)
# print("Intercept:" ,linear_regression.intercept_)


X = df['Likes'].values
Y = df['Views'].values

m = 0
b = 0
L = 0.0001
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