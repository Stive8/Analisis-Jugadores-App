import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import requests
from bs4 import BeautifulSoup

def entrenar_modelo(games_path="data/games.csv"):
    try:
        games = pd.read_csv(games_path, parse_dates=["date"])
        if not all(col in games.columns for col in ['date', 'home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals']):
            raise ValueError("Missing required columns in games.csv")

        # Crear dataframes locales y visitantes
        home_df = games[['date', 'home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals']].copy()
        home_df['team_id'] = home_df['home_club_id']
        home_df['opponent_id'] = home_df['away_club_id']
        home_df['is_home'] = 1
        home_df['goals_for'] = home_df['home_club_goals']
        home_df['goals_against'] = home_df['away_club_goals']

        away_df = games[['date', 'away_club_id', 'home_club_id', 'away_club_goals', 'home_club_goals']].copy()
        away_df['team_id'] = away_df['away_club_id']
        away_df['opponent_id'] = away_df['home_club_id']
        away_df['is_home'] = 0
        away_df['goals_for'] = away_df['away_club_goals']
        away_df['goals_against'] = away_df['home_club_goals']

        matches = pd.concat([home_df, away_df], ignore_index=True)
        matches = matches[['date', 'team_id', 'opponent_id', 'is_home', 'goals_for', 'goals_against']].sort_values(['team_id', 'date'])

        # Resultado del partido
        matches['result'] = matches.apply(lambda row: 'W' if row['goals_for'] > row['goals_against']
                                          else 'L' if row['goals_for'] < row['goals_against'] else 'D', axis=1)
        matches['goal_diff'] = matches['goals_for'] - matches['goals_against']

        # Rolling stats
        matches['goals_for_rolling'] = matches.groupby('team_id')['goals_for'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        matches['goals_against_rolling'] = matches.groupby('team_id')['goals_against'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        matches['goal_diff_rolling'] = matches.groupby('team_id')['goal_diff'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        result_map = {'W': 1, 'D': 0.5, 'L': 0}
        matches['result_code'] = matches['result'].map(result_map)
        matches['win_rate_rolling'] = matches.groupby('team_id')['result_code'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        # Limpiar NaNs
        matches = matches.dropna(subset=['goals_for_rolling', 'win_rate_rolling'])

        # Dividir local y visitante
        home_stats = matches[matches['is_home'] == 1].copy()
        away_stats = matches[matches['is_home'] == 0].copy()

        home_stats = home_stats.rename(columns={
            'team_id': 'home_team',
            'opponent_id': 'away_team',
            'goals_for_rolling': 'home_goals_avg',
            'goals_against_rolling': 'home_goals_conceded_avg',
            'goal_diff_rolling': 'home_goal_diff_avg',
            'win_rate_rolling': 'home_win_rate',
            'result': 'match_result'
        })

        away_stats = away_stats.rename(columns={
            'team_id': 'away_team',
            'opponent_id': 'home_team',
            'goals_for_rolling': 'away_goals_avg',
            'goals_against_rolling': 'away_goals_conceded_avg',
            'goal_diff_rolling': 'away_goal_diff_avg',
            'win_rate_rolling': 'away_win_rate',
            'result': 'match_result_away'
        })

        df_model = pd.merge(home_stats, away_stats, on=['date', 'home_team', 'away_team'], how='inner')

        if df_model.empty:
            raise ValueError("Merge resulted in an empty DataFrame. Check data alignment or matches.")

        if 'match_result' not in df_model.columns:
            raise KeyError("match_result column not found in df_model. Check merge and column renaming.")
        df_model['target'] = df_model['match_result'].map({'W': 1, 'D': 0, 'L': -1})

        numerical_columns = [
            'home_goals_avg', 'home_goals_conceded_avg', 'home_goal_diff_avg', 'home_win_rate',
            'away_goals_avg', 'away_goals_conceded_avg', 'away_goal_diff_avg', 'away_win_rate'
        ]
        X = df_model[numerical_columns]
        y = df_model['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        return model, matches

    except FileNotFoundError:
        raise FileNotFoundError("games.csv not found in data/ directory")
    except Exception as e:
        raise Exception(f"Error in entrenar_modelo: {e}")

def predecir_resultado(model, matches, home_id, away_id):
    last_home = matches[matches['team_id'] == home_id].sort_values('date').iloc[-1]
    last_away = matches[matches['team_id'] == away_id].sort_values('date').iloc[-1]

    nuevo_input = pd.DataFrame([{
        'home_goals_avg': last_home['goals_for_rolling'],
        'home_goals_conceded_avg': last_home['goals_against_rolling'],
        'home_goal_diff_avg': last_home['goal_diff_rolling'],
        'home_win_rate': last_home['win_rate_rolling'],
        'away_goals_avg': last_away['goals_for_rolling'],
        'away_goals_conceded_avg': last_away['goals_against_rolling'],
        'away_goal_diff_avg': last_away['goal_diff_rolling'],
        'away_win_rate': last_away['win_rate_rolling']
    }])

    pred = model.predict(nuevo_input)[0]
    prob = model.predict_proba(nuevo_input)[0]
    resultado = {1: "Gana el local", 0: "Empate", -1: "Gana el visitante"}
    return resultado[pred], prob

def obtener_rolling_stats_equipo(matches, team_id):
    equipo = matches[matches['team_id'] == team_id].sort_values('date').copy()
    return equipo[['date', 'goals_for_rolling', 'goals_against_rolling', 'goal_diff_rolling', 'win_rate_rolling']]

def obtener_nombre_equipo(club_id, clubs_df):
    fila = clubs_df[clubs_df["club_id"] == club_id]
    if not fila.empty:
        return fila.iloc[0]["name"]
    else:
        return f"Equipo {club_id}"

def obtener_url_escudo(club_url):
    try:
        # Hacer una solicitud HTTP a la página del club
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(club_url, headers=headers)
        response.raise_for_status()  # Lanzar excepción si hay error en la solicitud

        # Analizar el HTML con BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Buscar la imagen del escudo
        img_tag = soup.find('img', class_='tiny_wappen')
        if not img_tag:
            # Alternativa: buscar en el header del club
            img_tag = soup.find('div', class_='data-header__club-info').find('img') if soup.find('div', class_='data-header__club-info') else None

        if img_tag and 'src' in img_tag.attrs:
            img_url = img_tag['src']
            # Intentar obtener una versión de mayor resolución (reemplazar 'head' por 'big')
            img_url_high_res = img_url.replace('head', 'big') if 'head' in img_url else img_url
            # Asegurarse de que la URL sea absoluta
            if not img_url_high_res.startswith('http'):
                img_url_high_res = 'https://www.transfermarkt.co.uk' + img_url_high_res
            return img_url_high_res
        else:
            return None
    except Exception as e:
        print(f"Error al obtener la URL del escudo desde {club_url}: {e}")
        return None