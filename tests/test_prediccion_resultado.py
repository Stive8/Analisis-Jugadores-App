import sys
sys.path.append('C:/Users/Usuario/Documents/GitHub/Analisis-Jugadores-App')
import pandas as pd
import pytest
from unittest.mock import patch, Mock
import numpy as np
from utils.prediccion_resultado import entrenar_modelo, predecir_resultado, obtener_rolling_stats_equipo, obtener_nombre_equipo, obtener_url_escudo

# Datos simulados
MOCK_GAMES_DATA = {
    'date': ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01'],
    'home_club_id': [1, 2, 1, 3, 1, 2],
    'away_club_id': [2, 1, 3, 1, 2, 3],
    'home_club_goals': [2, 1, 3, 0, 2, 1],
    'away_club_goals': [1, 1, 0, 2, 1, 0],
    'team_id': [1, 2, 1, 3, 1, 2]
}

@patch('pandas.read_csv')
def test_entrenar_modelo_success(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame(MOCK_GAMES_DATA)
    model, matches = entrenar_modelo(games_path="data/games.csv")
    assert model is not None
    assert not matches.empty
    assert 'goals_for_rolling' in matches.columns

@patch('pandas.read_csv')
def test_entrenar_modelo_missing_columns(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({'date': ['2024-01-01'], 'home_club_id': [1]})
    with pytest.raises(Exception) as exc_info:
        entrenar_modelo(games_path="data/games.csv")
    assert str(exc_info.value) == "Error in entrenar_modelo: Missing required columns in games.csv"

@patch('pandas.read_csv')
def test_predecir_resultado_success(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame(MOCK_GAMES_DATA)
    model, matches = entrenar_modelo(games_path="data/games.csv")
    matches['date'] = pd.to_datetime(matches['date'])
    resultado, prob = predecir_resultado(model, matches, home_id=1, away_id=2)
    assert resultado in ['Gana el local', 'Empate', 'Gana el visitante']
    assert isinstance(prob, (list, tuple, np.ndarray))

def test_obtener_rolling_stats_equipo():
    # Simular el DataFrame procesado como en entrenar_modelo
    games = pd.DataFrame(MOCK_GAMES_DATA)
    games['date'] = pd.to_datetime(games['date'])
    
    # Crear home_df y away_df como en entrenar_modelo
    home_df = games[['date', 'home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals']].copy()
    home_df['team_id'] = home_df['home_club_id']
    home_df['opponent_id'] = home_df['away_club_id']
    home_df['is_home'] = 1
    home_df['goals_for'] = home_df['home_club_goals']
    home_df['goals_against'] = home_df['away_club_goals']

    away_df = games[['date', 'away_club_id', 'home_club_id', 'away_club_goals', 'home_club_goals']].copy()
    away_df['team_id'] = away_df['away_club_id']
    away_df['opponent_id'] = home_df['home_club_id']
    away_df['is_home'] = 0
    away_df['goals_for'] = away_df['away_club_goals']
    away_df['goals_against'] = away_df['home_club_goals']

    matches = pd.concat([home_df, away_df], ignore_index=True)
    matches = matches[['date', 'team_id', 'opponent_id', 'is_home', 'goals_for', 'goals_against']].sort_values(['team_id', 'date'])

    # Calcular resultados y estadísticas rodantes
    matches['result'] = matches.apply(lambda row: 'W' if row['goals_for'] > row['goals_against']
                                      else 'L' if row['goals_for'] < row['goals_against'] else 'D', axis=1)
    matches['goal_diff'] = matches['goals_for'] - matches['goals_against']
    matches['goals_for_rolling'] = matches.groupby('team_id')['goals_for'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    matches['goals_against_rolling'] = matches.groupby('team_id')['goals_against'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    matches['goal_diff_rolling'] = matches.groupby('team_id')['goal_diff'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    result_map = {'W': 1, 'D': 0.5, 'L': 0}
    matches['result_code'] = matches['result'].map(result_map)
    matches['win_rate_rolling'] = matches.groupby('team_id')['result_code'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

    # Llamar a la función con el DataFrame procesado
    stats = obtener_rolling_stats_equipo(matches, team_id=1)
    assert not stats.empty
    assert 'goals_for_rolling' in stats.columns

@patch('pandas.read_csv')
def test_obtener_nombre_equipo_success(mock_read_csv):
    clubs_df = pd.DataFrame({
        'club_id': [1],
        'name': ['Club A']
    })
    mock_read_csv.return_value = clubs_df
    name = obtener_nombre_equipo(club_id=1, clubs_df=clubs_df)
    assert name == 'Club A'

@patch('requests.get')
def test_obtener_url_escudo_success(mock_get):
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.text = '<html><img class="tiny_wappen" src="/images/head/1.png"></html>'
    mock_get.return_value = mock_response
    url = obtener_url_escudo('http://example.com/1')
    assert url == 'https://www.transfermarkt.co.uk/images/big/1.png'