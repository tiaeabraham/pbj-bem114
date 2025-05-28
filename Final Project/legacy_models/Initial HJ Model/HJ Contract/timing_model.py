# timing_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class TimingModel:
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()

    def prepare_data(self, df):
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        return df

    def train(self, df):
        X_train_timing = df[['DayOfWeek', 'Supply', 'Spread']].values
        y_train_timing = (df['Spread'].shift(-1) >
                          df['Spread']).astype(int).values
        X_train_scaled = self.scaler.fit_transform(X_train_timing)
        self.model.fit(X_train_scaled, y_train_timing)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X)
