import pandas as pd
import numpy as np
from datetime import datetime

print("Wczytywanie master.csv...")
# Zakładam, że plik znajduje się w folderze data/csv
df = pd.read_csv('data/csv/master.csv')

# 1. CZYSZCZENIE I FORMATOWANIE DAT
print("Czyszczenie danych i konwersja dat...")
# Zmieniamy typy dat na datetime
df['event_date'] = pd.to_datetime(df['event_date'])

# W danych z UFC Stats często brakuje pełnej daty urodzenia, czasami jest '--' lub puste.
# Musimy obsłużyć błędy podczas konwersji
df['r_dob'] = pd.to_datetime(df['r_dob'], errors='coerce')
df['b_dob'] = pd.to_datetime(df['b_dob'], errors='coerce')

# 2. WYLICZANIE WIEKU W DNIU WALKI
print("Wyliczanie wieku w dniu walki...")
# Zwracamy wiek w latach, jeśli brakuje daty urodzenia, wstawiamy NaN (XGBoost sobie z tym poradzi)
df['r_age'] = (df['event_date'] - df['r_dob']).dt.days / 365.25
df['b_age'] = (df['event_date'] - df['b_dob']).dt.days / 365.25

# Wiek Delta (na minus = R jest młodszy, na plus = R jest starszy)
df['age_diff'] = df['r_age'] - df['b_age']

# 3. WYLICZANIE PRZEWAGI FIZYCZNEJ (Zasięg i Wzrost)
print("Konwersja miar z systemu imperialnego na centymetry...")


# Funkcja pomocnicza do konwersji wzrostu z formatu "6' 3"" na cale, a potem cm
def parse_height_to_cm(height_str):
    if pd.isna(height_str) or height_str == '--':
        return np.nan
    try:
        # Usuwamy cudzysłowy i dzielimy na stopy i cale
        clean_str = height_str.replace('"', '').strip()
        parts = clean_str.split("' ")
        feet = float(parts[0])
        inches = float(parts[1]) if len(parts) > 1 else 0.0
        total_inches = (feet * 12) + inches
        return total_inches * 2.54
    except:
        return np.nan


df['r_height_cm'] = df['r_height'].apply(parse_height_to_cm)
df['b_height_cm'] = df['b_height'].apply(parse_height_to_cm)

# Zasięg (reach) w UFCStats jest podany w calach, jeśli w ogóle jest.
df['r_reach_cm'] = pd.to_numeric(df['r_reach_inches'], errors='coerce') * 2.54
df['b_reach_cm'] = pd.to_numeric(df['b_reach_inches'], errors='coerce') * 2.54

# Delty fizyczne
df['height_diff'] = df['r_height_cm'] - df['b_height_cm']
df['reach_diff'] = df['r_reach_cm'] - df['b_reach_cm']

# 4. KODOWANIE POSTAWY (Stance) I WALKI O PAS (Title Fight)
print("Kodowanie zmiennych kategorycznych...")
df['title_fight'] = df['title_fight'].astype(int)

# Konwersja postawy na wartości liczbowe lub flagi
# Ponieważ chcemy podać modelowi jednoznaczny zestaw różnic:
df['r_is_southpaw'] = (df['r_stance'] == 'Southpaw').astype(int)
df['b_is_southpaw'] = (df['b_stance'] == 'Southpaw').astype(int)
# Czy walczą przeciwne postawy? (1 = tak, 0 = nie)
df['opposite_stance'] = (df['r_is_southpaw'] != df['b_is_southpaw']).astype(int)

# 5. OBLICZANIE HISTORII ZAWODNIKA (Unikamy Data Leakage)
print("Budowanie historii walk (Cumulative Stats)...")
# Sortujemy chronologicznie całą bazę! (Bardzo ważne!)
df = df.sort_values(by='event_date', ascending=True).reset_index(drop=True)

# Tworzymy słowniki, które będą trzymać tymczasowe statystyki zawodników W MIARĘ JAK idziemy przez czas.
fighter_stats = {}


def get_fighter_stats(f_id):
    if f_id not in fighter_stats:
        fighter_stats[f_id] = {
            'wins': 0, 'losses': 0, 'draws': 0,
            'current_win_streak': 0, 'current_lose_streak': 0,
            'total_time_seconds': 0,
            'sig_strikes_landed': 0, 'sig_strikes_absorbed': 0,
            'takedowns_landed': 0, 'takedowns_attempted': 0
        }
    return fighter_stats[f_id].copy()  # Zwracamy kopię stanu NA MOMENT PRZED walką!


def update_fighter_stats(f_id, result, time_spent, sig_landed, sig_absorbed, td_landed, td_attempted):
    stats = fighter_stats[f_id]

    if result == 'win':
        stats['wins'] += 1
        stats['current_win_streak'] += 1
        stats['current_lose_streak'] = 0
    elif result == 'loss':
        stats['losses'] += 1
        stats['current_lose_streak'] += 1
        stats['current_win_streak'] = 0
    else:
        stats['draws'] += 1
        stats['current_win_streak'] = 0
        stats['current_lose_streak'] = 0

    stats['total_time_seconds'] += time_spent
    stats['sig_strikes_landed'] += sig_landed
    stats['sig_strikes_absorbed'] += sig_absorbed
    stats['takedowns_landed'] += td_landed
    stats['takedowns_attempted'] += td_attempted


# Inicjalizacja nowych kolumn dla stanu wiedzy (Cumulative)
cumulative_features = []


# Musimy wyciągnąć informacje o czasie trwania walki (w sekundach).
# Format czasu to np. "2:02", musimy to przekonwertować, uwzględniając rundy.
def get_fight_duration_seconds(round_no, finish_time):
    try:
        minutes, seconds = finish_time.split(':')
        # Zakładamy 5 minut (300 sek) na rundę, co jest standardem
        total_seconds = ((int(round_no) - 1) * 300) + (int(minutes) * 60) + int(seconds)
        return total_seconds
    except:
        return 0


# Przechodzimy przez każdą walkę wiersz po wierszu
for idx, row in df.iterrows():
    # Pobieramy stan NA MOMENT PRZED walką
    r_stats_before = get_fighter_stats(row['r_fighter_id'])
    b_stats_before = get_fighter_stats(row['b_fighter_id'])

    # Wyliczamy statystyki uderzeń na minutę do tego momentu
    r_mins_fought = r_stats_before['total_time_seconds'] / 60
    b_mins_fought = b_stats_before['total_time_seconds'] / 60

    r_slpm_cum = (r_stats_before['sig_strikes_landed'] / r_mins_fought) if r_mins_fought > 0 else 0
    r_sapm_cum = (r_stats_before['sig_strikes_absorbed'] / r_mins_fought) if r_mins_fought > 0 else 0
    b_slpm_cum = (b_stats_before['sig_strikes_landed'] / b_mins_fought) if b_mins_fought > 0 else 0
    b_sapm_cum = (b_stats_before['sig_strikes_absorbed'] / b_mins_fought) if b_mins_fought > 0 else 0

    cumulative_features.append({
        'r_win_streak': r_stats_before['current_win_streak'],
        'b_win_streak': b_stats_before['current_win_streak'],
        'r_lose_streak': r_stats_before['current_lose_streak'],
        'b_lose_streak': b_stats_before['current_lose_streak'],
        'r_total_fights': r_stats_before['wins'] + r_stats_before['losses'] + r_stats_before['draws'],
        'b_total_fights': b_stats_before['wins'] + b_stats_before['losses'] + b_stats_before['draws'],
        'r_slpm_cum': r_slpm_cum,
        'r_sapm_cum': r_sapm_cum,
        'b_slpm_cum': b_slpm_cum,
        'b_sapm_cum': b_sapm_cum,
        'slpm_diff': r_slpm_cum - b_slpm_cum,
        'sapm_diff': r_sapm_cum - b_sapm_cum
    })

    # PO ZAPISANIU STANU AKTUALIZUJEMY STATYSTYKI NA BAZIE WYNIKU TEJ WALKI
    duration = get_fight_duration_seconds(row['finish_round'], row['finish_time'])

    # R_id win = R win, B loss
    if row['result_status'] == 'win':
        if row['winner_id'] == row['r_fighter_id']:
            r_res, b_res = 'win', 'loss'
        elif row['winner_id'] == row['b_fighter_id']:
            r_res, b_res = 'loss', 'win'
        else:
            r_res, b_res = 'draw', 'draw'
    else:
        r_res, b_res = 'draw', 'draw'  # NC, Draws itp.

    update_fighter_stats(
        row['r_fighter_id'], r_res, duration,
        row['r_total_sig_landed'], row['b_total_sig_landed'],  # Absorbed dla R to landed dla B
        row['r_total_td_success'], row['r_total_td_atmp']
    )
    update_fighter_stats(
        row['b_fighter_id'], b_res, duration,
        row['b_total_sig_landed'], row['r_total_sig_landed'],
        row['b_total_td_success'], row['b_total_td_atmp']
    )

# Doklejamy wyliczone statystyki historyczne do głównego DataFrame
cum_df = pd.DataFrame(cumulative_features)
df = pd.concat([df, cum_df], axis=1)

# 6. DEFINIOWANIE ZMIENNEJ ZALEŻNEJ (TARGET) I ODRZUCENIE REMISÓW/NC
print("Filtrowanie wyników i tworzenie targetu...")
# Do modelu bierzemy tylko walki, gdzie był jasny zwycięzca
df = df[df['result_status'] == 'win'].copy()
# Target: 1 jeśli wygrał zawodnik Red (R), 0 jeśli Blue (B)
df['target'] = (df['winner_id'] == df['r_fighter_id']).astype(int)

# 7. ZAPIS DO PLIKU ML_READY
print("Wybieranie kolumn wejściowych i zapis...")
# Wybieramy TYLKO te kolumny, które model może bezpiecznie widzieć (bez wycieków danych)
ml_features = [
    'fight_id', 'event_date', 'weight_class', 'title_fight',
    'age_diff', 'height_diff', 'reach_diff', 'opposite_stance',
    'r_total_fights', 'b_total_fights',
    'r_win_streak', 'b_win_streak', 'r_lose_streak', 'b_lose_streak',
    'slpm_diff', 'sapm_diff',
    'target'
]

df_ml = df[ml_features].dropna()  # Usuwamy wiersze gdzie brakuje krytycznych danych fizycznych
df_ml.to_csv('ml_ready.csv', index=False)
print(f"Gotowe! Zapisano plik data/csv/ml_ready.csv o wymiarach: {df_ml.shape}")