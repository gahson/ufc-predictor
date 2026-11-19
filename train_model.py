import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

print("Wczytywanie danych...")
df = pd.read_csv('data/csv/ml_ready.csv')

# Upewniamy się, że dane są posortowane chronologicznie
df['event_date'] = pd.to_datetime(df['event_date'])
df = df.sort_values('event_date').reset_index(drop=True)

# 1. PRZYGOTOWANIE ZMIENNYCH
# Kategoria wagowa to tekst (np. Heavyweight), model potrzebuje liczb.
# Zamieniamy to na tzw. One-Hot Encoding (tworzy flagi 0/1 dla każdej wagi).
df = pd.get_dummies(df, columns=['weight_class'])

# Oddzielamy cechy (X) od wyniku (y)
# Usuwamy ID walki i datę, bo to nie są cechy, na których model ma się uczyć
X = df.drop(columns=['fight_id', 'event_date', 'target'])
y = df['target']

# 2. CHRONOLOGICZNY PODZIAŁ DANYCH (Train/Test Split)
# Trenujemy na pierwszych 80% walk (historia), testujemy na ostatnich 20% (najnowsze).
# NIGDY nie używaj zwykłego train_test_split (losowego) w sporcie, bo wymieszasz ramy czasowe!
split_index = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

test_start_date = df.iloc[split_index]['event_date'].date()
print(f"Dane treningowe: {len(X_train)} walk.")
print(f"Dane testowe: {len(X_test)} walk (od {test_start_date}).")

# 3. TRENOWANIE MODELU XGBOOST
print("\nTrenowanie modelu...")
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# 4. OCENA SKUTECZNOŚCI
print("\nEwaluacja na zbiorze testowym (przyszłych walkach):")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy (Dokładność): {accuracy * 100:.2f}%")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nSzczegółowy raport:")
print(classification_report(y_test, y_pred))

# 5. WAŻNOŚĆ CECH (Feature Importance)
# Sprawdzamy, na co model zwraca największą uwagę przy przewidywaniu
importance = pd.DataFrame({
    'Cecha': X.columns,
    'Ważność': model.feature_importances_
}).sort_values(by='Ważność', ascending=False)

print("\nTop 10 najważniejszych statystyk decydujących o wygranej:")
print(importance.head(10).to_string(index=False))

joblib.dump(model, 'ufc_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
print("Zapisano model i kolumny!")