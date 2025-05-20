import sys
sys.path.append('C:/Users/Usuario/Documents/GitHub/Analisis-Jugadores-App')
import pandas as pd
import pytest
from unittest.mock import patch
from utils.procesado import plot_predicciones_arima

# Datos de prueba simulados
MOCK_APPEARANCES_DATA = {
    'player_id': [1, 1, 1],
    'goals': [2, 1, 0],
    'assists': [1, 0, 2],
    'date': ['2024-01-01', '2024-02-01', '2024-03-01']
}
MOCK_PLAYERS_DATA = {
    'player_id': [1],
    'name': ['Player A']
}

@patch('pandas.read_csv')
def test_plot_predicciones_arima_success(mock_read_csv):
    mock_read_csv.side_effect = [
        pd.DataFrame(MOCK_APPEARANCES_DATA),
        pd.DataFrame(MOCK_PLAYERS_DATA)
    ]
    name, stats, fig_goals, fig_assists = plot_predicciones_arima(player_id=1, years_back=2, appearances_path="data/appearances.csv")
    assert name == 'Player A'
    assert stats['total_goals'] == 3
    assert fig_goals is not None
    assert fig_assists is not None

@patch('pandas.read_csv')
def test_plot_predicciones_arima_no_data(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame(columns=['player_id', 'goals', 'assists', 'date'])
    result = plot_predicciones_arima(player_id=1, years_back=2, appearances_path="data/appearances.csv")
    assert result is None

@patch('pandas.read_csv')
def test_plot_predicciones_arima_file_not_found(mock_read_csv):
    mock_read_csv.side_effect = FileNotFoundError
    result = plot_predicciones_arima(player_id=1, years_back=2, appearances_path="data/appearances.csv")
    assert result is None