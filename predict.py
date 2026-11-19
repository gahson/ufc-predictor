import pandas as pd
import numpy as np
import joblib
from datetime import datetime

model = joblib.load('ufc_model.pkl')
model_columns = joblib.load('model_columns.pkl')
master_df = pd.read_csv('data/csv/master.csv')
master_df['event_date'] = pd.to_datetime(master_df['event_date'])


def parse_height_to_cm(h):
    if pd.isna(h) or h == '--': return np.nan
    try:
        parts = str(h).replace('"', '').strip().split("' ")
        return ((float(parts[0]) * 12) + (float(parts[1]) if len(parts) > 1 else 0.0)) * 2.54
    except:
        return np.nan


def get_duration(rnd, time_str):
    try:
        m, s = str(time_str).split(':')
        return ((int(rnd) - 1) * 300) + (int(m) * 60) + int(s)
    except:
        return 0


def get_fighter_profile(query):
    # Szukamy zawodnika w obu narożnikach w master.csv
    mask_r = master_df['r_fighter_name'].str.contains(query, case=False, na=False)
    mask_b = master_df['b_fighter_name'].str.contains(query, case=False, na=False)
    matches = master_df[mask_r | mask_b].sort_values('event_date', ascending=False)

    if matches.empty:
        raise ValueError(f"Brak '{query}' w master.csv (zawodnik nie ma jeszcze walk w bazie UFC).")

    # Bierzemy najnowszą walkę, żeby wyciągnąć ID i aktualne dane fizyczne
    latest = matches.iloc[0]
    if pd.notna(latest['r_fighter_name']) and query.lower() in str(latest['r_fighter_name']).lower():
        prefix = 'r_'
    else:
        prefix = 'b_'

    f_id = latest[f'{prefix}fighter_id']
    name = latest[f'{prefix}fighter_name']
    dob = pd.to_datetime(latest[f'{prefix}dob'], errors='coerce')
    height_cm = parse_height_to_cm(latest[f'{prefix}height'])
    reach_cm = pd.to_numeric(latest[f'{prefix}reach_inches'], errors='coerce') * 2.54
    is_southpaw = 1 if latest[f'{prefix}stance'] == 'Southpaw' else 0

    # Filtrujemy całą historię tego konkretnego fighter_id chronologicznie
    fights = master_df[(master_df['r_fighter_id'] == f_id) | (master_df['b_fighter_id'] == f_id)].sort_values(
        'event_date')

    wins, losses, draws, win_streak, lose_streak = 0, 0, 0, 0, 0
    total_sec, sig_landed, sig_absorbed = 0, 0, 0

    for _, f in fights.iterrows():
        is_red = (f['r_fighter_id'] == f_id)
        if f['result_status'] == 'win':
            if f['winner_id'] == f_id:
                wins += 1;
                win_streak += 1;
                lose_streak = 0
            else:
                losses += 1;
                lose_streak += 1;
                win_streak = 0
        else:
            draws += 1;
            win_streak = 0;
            lose_streak = 0

        total_sec += get_duration(f['finish_round'], f['finish_time'])
        sig_landed += f['r_total_sig_landed'] if is_red else f['b_total_sig_landed']
        sig_absorbed += f['b_total_sig_landed'] if is_red else f['r_total_sig_landed']

    mins = total_sec / 60
    return {
        'name': name,
        'dob': dob,
        'height_cm': height_cm,
        'reach_cm': reach_cm,
        'is_southpaw': is_southpaw,
        'total_fights': wins + losses + draws,
        'win_streak': win_streak,
        'lose_streak': lose_streak,
        'slpm': (sig_landed / mins) if mins > 0 else 0,
        'sapm': (sig_absorbed / mins) if mins > 0 else 0
    }


def run_single_prediction(r, b, weight_class, title_fight, fight_date):
    r_age = (fight_date - r['dob']).days / 365.25
    b_age = (fight_date - b['dob']).days / 365.25

    features = {
        'title_fight': title_fight,
        'age_diff': r_age - b_age,
        'height_diff': r['height_cm'] - b['height_cm'],
        'reach_diff': r['reach_cm'] - b['reach_cm'],
        'opposite_stance': 1 if r['is_southpaw'] != b['is_southpaw'] else 0,
        'r_total_fights': r['total_fights'],
        'b_total_fights': b['total_fights'],
        'r_win_streak': r['win_streak'],
        'b_win_streak': b['win_streak'],
        'r_lose_streak': r['lose_streak'],
        'b_lose_streak': b['lose_streak'],
        'slpm_diff': r['slpm'] - b['slpm'],
        'sapm_diff': r['sapm'] - b['sapm'],
    }

    df_in = pd.DataFrame([features])
    wc_col = f"weight_class_{weight_class}"
    for col in model_columns:
        if col not in df_in.columns:
            df_in[col] = 1 if col == wc_col else 0

    return model.predict_proba(df_in[model_columns])[0][1]


def predict(fighter_a, fighter_b, weight_class="Welterweight", title_fight=0):
    today = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
    a = get_fighter_profile(fighter_a)
    b = get_fighter_profile(fighter_b)

    # Uśrednianie obu narożników (neutralizacja Red Corner Bias)
    prob_a_as_red = run_single_prediction(a, b, weight_class, title_fight, today)
    prob_b_as_red = run_single_prediction(b, a, weight_class, title_fight, today)

    prob_a = (prob_a_as_red + (1 - prob_b_as_red)) / 2
    prob_b = 1 - prob_a

    print(f"\n🥊 {a['name']} vs {b['name']} ({weight_class})")
    print(
        f"   {a['name']}: {prob_a * 100:.1f}% (uczciwy kurs: {1 / prob_a:.2f}) | Walki w UFC: {a['total_fights']}, Streak: W{a['win_streak']}/L{a['lose_streak']}")
    print(
        f"   {b['name']}: {prob_b * 100:.1f}% (uczciwy kurs: {1 / prob_b:.2f}) | Walki w UFC: {b['total_fights']}, Streak: W{b['win_streak']}/L{b['lose_streak']}")


if __name__ == "__main__":
    predict("Machado", "Amosov", weight_class="Welterweight", title_fight=0)