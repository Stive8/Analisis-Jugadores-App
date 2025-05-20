import sys
sys.path.append('C:/Users/Usuario/Documents/GitHub/Analisis-Jugadores-App')
import pandas as pd
import pytest
from unittest.mock import patch
from utils.minuto import sugerencias_por_intervalo

# Datos de prueba simulados
MOCK_PLAYERS_DATA = {
    'player_id': [1, 2],
    'name': ['Player A', 'Player B'],
    'position': ['Attack', 'Defense']
}
MOCK_EVENTS_DATA = {
    'type': ['Goals', 'Cards', 'Goals', 'Cards'],
    'player_id': [1, 1, 2, 2],
    'minute': [5, 15, 25, 35],
    'game_id': [100, 100, 101, 101]
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
    assert len(stats['cluster'].unique()) <= 3  # Máximo 3 clusters
    assert stats[stats['player_id'] == 1]['goals'].iloc[0] == 1  # 1 gol en el intervalo 1-10

@patch('pandas.read_csv')
def test_sugerencias_por_intervalo_empty(mock_read_csv):
    mock_read_csv.side_effect = [
        pd.DataFrame(MOCK_PLAYERS_DATA),
        pd.DataFrame(MOCK_EVENTS_DATA[MOCK_EVENTS_DATA['minute'] > 10])  # Sin datos en 1-10
    ]
    stats, X_scaled, kmeans = sugerencias_por_intervalo(intervalo="1-10", data_path="data")
    assert stats is None
    assert X_scaled is None
    assert kmeans is None

@patch('pandas.read_csv')
def test_sugerencias_por_intervalo_invalid_minute(mock_read_csv):
    mock_events = MOCK_EVENTS_DATA.copy()
    mock_events['minute'] = ['invalid', '15', '25', '35']
    mock_read_csv.side_effect = [
        pd.DataFrame(MOCK_PLAYERS_DATA),
        pd.DataFrame(mock_events)
    ]
    with pytest.raises(ValueError):
        sugerencias_por_intervalo(intervalo="1-10", data_path="data")