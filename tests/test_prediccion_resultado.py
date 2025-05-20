import sys
sys.path.append('C:/Users/Usuario/Documents/GitHub/Analisis-Jugadores-App')
import pandas as pd
import pytest
from unittest.mock import patch
from utils.prediccion_resultado import entrenar_modelo, predecir_resultado, obtener_rolling_stats_equipo, obtener_nombre_equipo, obtener_url_escudo

# Datos de prueba simulados
MOCK_GAMES_DATA = {
    'date': ['2024-01-01', '2024-02-01', '2024-03-01'],
    'home_club_id': [1, 2, 1],
    'away_club_id': [2, 1, 3],
    'home_club_goals': [2, 1, 3],
    'away_club_goals': [1, 2, 0]
}
MOCK_CLUBS_DATA = {
    'club_id': [1, 2, 3],
    'name': ['Team A', 'Team B', 'Team C'],
    'url': ['http://example.com/1', 'http://example.com/2', 'http://example.com/3']
}

@patch('pandas.read_csv')
def test_entrenar_modelo_success(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame(MOCK_GAMES_DATA)
    model, matches = entrenar_modelo(games_path="data/games.csv")
    assert model is not None
    assert not matches.empty
    assert 'result' in matches.columns

@patch('pandas.read_csv')
def test_entrenar_modelo_missing_columns(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({'date': ['2024-01-01'], 'home_club_id': [1]})  # Sin 'home_club_goals'
    with pytest.raises(ValueError):
        entrenar_modelo(games_path="data/games.csv")

@patch('pandas.read_csv')
def test_predecir_resultado_success(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame(MOCK_GAMES_DATA)
    model, matches = entrenar_modelo(games_path="data/games.csv")
    resultado, prob = predecir_resultado(model, matches, home_id=1, away_id=2)
    assert resultado in ['Gana el local', 'Empate', 'Gana el visitante']
    assert len(prob) == 3  # Probabilidades para 3 clases

def test_obtener_rolling_stats_equipo():
    matches = pd.DataFrame(MOCK_GAMES_DATA)
    stats = obtener_rolling_stats_equipo(matches, team_id=1)
    assert not stats.empty
    assert 'goals_for_rolling' in stats.columns

@patch('pandas.read_csv')
def test_obtener_nombre_equipo_success(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame(MOCK_CLUBS_DATA)
    name = obtener_nombre_equipo(club_id=1, clubs_df=pd.DataFrame(MOCK_CLUBS_DATA))
    assert name == 'Team A'

@patch('requests.get')
def test_obtener_url_escudo_success(mock_get):
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.text = '<html><img class="tiny_wappen" src="/images/head/1.png"></html>'
    mock_get.return_value = mock_response
    url = obtener_url_escudo('http://example.com/1')
    assert url == 'https://www.transfermarkt.co.uk/images/big/1.png'