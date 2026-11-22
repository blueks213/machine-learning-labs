import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Crime_Data_from_2020_to_Present.csv')

df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p')

df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4)

df['DATETIME'] = pd.to_datetime(df['DATE OCC'].dt.strftime('%Y-%m-%d') + ' ' +
                                df['TIME OCC'].str[:2] + ':' + df['TIME OCC'].str[2:4])

df['year'] = df['DATETIME'].dt.year
df['month'] = df['DATETIME'].dt.month
df['day'] = df['DATETIME'].dt.day
df['hour'] = df['DATETIME'].dt.hour
df['weekday'] = df['DATETIME'].dt.weekday

df = df.dropna(subset=['LAT', 'LON'])

region_crime = df.groupby('AREA NAME').size().reset_index(name='crime_count')

features = df.groupby('AREA NAME').agg({
    'LAT': 'mean',
    'LON': 'mean',
    'hour': 'mean',
    'weekday': 'mean',
    'month': 'nunique',
    'Crm Cd': 'nunique'
}).reset_index()

features = features.merge(region_crime, on='AREA NAME')

X = features[['LAT', 'LON', 'hour', 'weekday', 'month', 'Crm Cd']]
y = features['crime_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'LinearRegression': LinearRegression(),
    'SVR': SVR()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f' {name} ')
    print('MSE:', mean_squared_error(y_test, y_pred))
    print('R2:', r2_score(y_test, y_pred))
    print()