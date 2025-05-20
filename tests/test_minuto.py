import sys
sys.path.append('C:/Users/Usuario/Documents/GitHub/Analisis-Jugadores-App')
import pandas as pd
import pytest
from unittest.mock import patch
from utils.minuto import sugerencias_por_intervalo

# Datos simulados
MOCK_PLAYERS_DATA = {
    'player_id': [1, 2, 3],
    'name': ['Player A', 'Player B', 'Player C'],
    'position': ['Attack', 'Defense', 'Midfield']
}

MOCK_EVENTS_DATA = {
    'type': ['Goals', 'Cards', 'Goals', 'Cards', 'Goals', 'Cards', 'Goals'],
    'player_id': [1, 1, 2, 2, 3, 3, 1],
    'minute': [5, 7, 3, 8, 9, 6, 4],
    'game_id': [100, 100, 101, 101, 102, 102, 103]
}

@patch('pandas.read_csv')
def test_sugerencias_por_intervalo_success(mock_read_csv):
    mock_read_csv.side_effect = [
        pd.DataFrame(MOCK_PLAYERS_DATA),
        pd.DataFrame(MOCK_EVENTS_DATA)
    ]
    stats, X_scaled, kmeans = sugerencias_por_intervalo(intervalo="1-10", data_path="data")
    assert not stats.empty
    assert 'cluster' in stats.columns
    assert len(stats['cluster'].unique()) <= 3
    assert stats[stats['player_id'] == 1]['goals'].sum() >= 1

@patch('pandas.read_csv')
def test_sugerencias_por_intervalo_empty(mock_read_csv):
    mock_events = pd.DataFrame({
        'type': ['Goals', 'Cards'],
        'player_id': [1, 2],
        'minute': [15, 25],
        'game_id': [100, 101]
    })
    mock_read_csv.side_effect = [
        pd.DataFrame(MOCK_PLAYERS_DATA),
        mock_events
    ]
    stats, X_scaled, kmeans = sugerencias_por_intervalo(intervalo="1-10", data_path="data")
    assert stats is None
    assert X_scaled is None
    assert kmeans is None

