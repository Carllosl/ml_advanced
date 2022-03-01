import numpy as np
import pandas as pd
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from warnings import simplefilter
from sklearn import metrics
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def date_to_onehot(df):
    df['Дата накладной'] = pd.to_datetime(df['Дата накладной'], dayfirst=True)
    df['Дата накладной'] = df['Дата накладной'].dt.month
    onehot_date = pd.get_dummies(df['Дата накладной'])
    df = df.drop('Дата накладной', axis=1)
    df = df.join(onehot_date)
    return df


def channel_to_onehot(df, df_mapping):
    mapping = {row['Канал']: row['Направление продаж'] for index, row in df_mapping.iterrows()}
    df = df.replace(mapping)
    onehot_channel = pd.get_dummies(df['Канал'])
    df = df.drop('Канал', axis=1)
    df = df.join(onehot_channel)
    return df


def prepare_df_discount(df_discount):
    df_discount = df_discount.drop('Year', axis=1)
    df_discount['Код товара'] = df_discount['Код товара'].str.replace('SKU_', '')
    df_discount['Код товара'] = pd.to_numeric(df_discount['Код товара'])
    return df_discount


def regression(df, x_col):
    X = df[x_col]
    y = df['Продажи, шт']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coeff'])
#    y_pred = regressor.predict(X_test)
#    res_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    inter = regressor.intercept_
    return coeff_df, inter


def prediction(df_discount, coeff, inter, el, res_df):
    df_discount = df_discount.loc[df_discount['Код товара'] == el]
    flag = df_discount.empty
    for i in 7,8,9:
        flag2 = False
        if not flag:
            df_dic = df_discount.loc[df_discount['Month'] == i]
            flag2 = df_dic.empty
        for opt in 0,1:
            disc = 0
            if opt == 0: roz = 1
            else: roz = 0
            if not flag and not flag2:
                for index,row in df_dic.iterrows():
                    if row['Направление продаж'] == 'Оптовая торговля':
                        if opt == 1:
                            disc = row['Скидка']
                    else:
                        if opt == 0:
                            disc = row['Скидка']
            num = inter + coeff.loc[i]['Coeff'] + coeff.loc['Оптовая торговля']['Coeff']*opt + coeff.loc['Розничная торговля']['Coeff']*roz + coeff.loc['скидка']['Coeff']*disc
            if opt == 1: napr = 'Оптовая торговля'
            else: napr = 'Розничная торговля'
            line = {'Код товара': el, 'Месяц': i, 'Направление продаж': napr, 'Продажи, шт': round(num)}
            res_df = res_df.append(line, ignore_index=True)
    return(res_df)


df = pd.read_csv('Shipments_by_PO.csv')
df_mapping = pd.read_csv('Mapping.csv')
df_discount = pd.read_excel('Forecast_of_discounts.xlsx', sheet_name='Sheet1')
#print(df_discount.head())
df = date_to_onehot(df)
df = channel_to_onehot(df, df_mapping)
code_list = pd.unique(df['Код товара'])
x_col = list(df)
del x_col[0]
del x_col[0]
df_discount = prepare_df_discount(df_discount)
res_df = pd.DataFrame()
for el in code_list:
    if el != 324:
        product_df = df.loc[df['Код товара'] == el]
        coeff, inter = regression(product_df, x_col)
        res_df = prediction(df_discount, coeff, inter, el, res_df)
print(res_df)
res_df.to_csv('Result.csv')

