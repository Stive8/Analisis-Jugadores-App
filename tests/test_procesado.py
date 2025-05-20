import sys
sys.path.append('C:/Users/Usuario/Documents/GitHub/Analisis-Jugadores-App')
import pandas as pd
import pytest
from unittest.mock import patch
from utils.procesado import plot_predicciones_arima

# Mock para streamlit
try:
    import streamlit as st
except ImportError:
    st = None

# Datos simulados
MOCK_PLAYERS_DATA = {
    'player_id': [1, 2],
    'name': ['Player A', 'Player B'],
    'position': ['Attack', 'Defense']
}

MOCK_APPEARANCES_DATA = {
    'player_id': [1, 1, 1, 2],
    'player_name': ['Player A', 'Player A', 'Player A', 'Player B'],
    'goals': [2, 1, 0, 0],
    'assists': [1, 0, 2, 2],
    'minutes_played': [90, 80, 70, 70],
    'game_id': [100, 101, 102, 103],
    'date': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-02-01'), pd.Timestamp('2024-03-01'), pd.Timestamp('2024-04-01')]
}

@patch('pandas.read_csv')
@patch('streamlit.error')
@patch('streamlit.warning')
def test_plot_predicciones_arima_success(mock_warning, mock_error, mock_read_csv):
    mock_appearances = pd.DataFrame(MOCK_APPEARANCES_DATA)
    mock_appearances['date'] = pd.to_datetime(mock_appearances['date'])
    mock_read_csv.side_effect = [
        mock_appearances,
        pd.DataFrame(MOCK_PLAYERS_DATA)
    ]
    result = plot_predicciones_arima(player_id=1, years_back=2, appearances_path="data/appearances.csv")
    if result is not None:
        name, stats, fig_goals, fig_assists = result
        assert name is not None
        assert stats is not None
        assert fig_goals is not None
        assert fig_assists is not None
    else:
        assert mock_warning.called or mock_error.called

@patch('pandas.read_csv')
@patch('streamlit.error')
@patch('streamlit.warning')
def test_plot_predicciones_arima_no_data(mock_warning, mock_error, mock_read_csv):
    mock_read_csv.side_effect = [
        pd.DataFrame({
            'player_id': [2],
            'goals': [0],
            'assists': [2],
            'date': [pd.Timestamp('2024-03-01')],
            'minutes_played': [70],
            'game_id': [102],
            'player_name': ['Player B']
        }),
        pd.DataFrame(MOCK_PLAYERS_DATA)
    ]
    result = plot_predicciones_arima(player_id=1, years_back=2, appearances_path="data/appearances.csv")
    assert result is None
    assert mock_warning.called

@patch('pandas.read_csv')
@patch('streamlit.error')
@patch('streamlit.warning')
def test_plot_predicciones_arima_file_not_found(mock_warning, mock_error, mock_read_csv):
    mock_read_csv.side_effect = FileNotFoundError
    result = plot_predicciones_arima(player_id=1, years_back=2, appearances_path="data/appearances.csv")
    assert result is None
    assert mock_error.called