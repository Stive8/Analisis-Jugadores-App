import sys
sys.path.append('C:/Users/Usuario/Documents/GitHub/Analisis-Jugadores-App')
import pandas as pd
import pytest
from unittest.mock import patch
from utils.clustering import cargar_datos_cluster, generar_clusters, recomendar

# Datos simulados
MOCK_PLAYERS_DATA = {
    'player_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'name': ['Player A', 'Player B', 'Player C', 'Player D', 'Player E', 'Player F', 'Player G', 'Player H', 'Player I', 'Player J'],
    'position': ['Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack']
}

MOCK_APPEARANCES_DATA = {
    'player_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
    'player_name': ['Player A', 'Player A', 'Player B', 'Player B', 'Player C', 'Player C', 'Player D', 'Player D', 'Player E', 'Player E',
                    'Player F', 'Player F', 'Player G', 'Player G', 'Player H', 'Player H', 'Player I', 'Player I', 'Player J', 'Player J'],
    'goals': [2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1],
    'assists': [1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0],
    'minutes_played': [90, 80, 70, 90, 80, 70, 90, 80, 70, 90, 80, 70, 90, 80, 70, 90, 80, 70, 90, 80],
    'game_id': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
    'date': ['2024-01-01', '2024-02-01', '2024-03-01', '2024-01-01', '2024-02-01', '2024-03-01', '2024-01-01', '2024-02-01', '2024-03-01',
             '2024-01-01', '2024-02-01', '2024-03-01', '2024-01-01', '2024-02-01', '2024-03-01', '2024-01-01', '2024-02-01', '2024-03-01',
             '2024-01-01', '2024-02-01']
}

@patch('pandas.read_csv')
def test_cargar_datos_cluster_success(mock_read_csv):
    mock_read_csv.side_effect = [
        pd.DataFrame(MOCK_PLAYERS_DATA),
        pd.DataFrame(MOCK_APPEARANCES_DATA)
    ]
    stats = cargar_datos_cluster(tipo_posiciones=['Attack'], data_path='data')
    assert not stats.empty
    assert 'goals_per_game' in stats.columns
    assert 'assists_per_game' in stats.columns
    assert len(stats) == 10

@patch('pandas.read_csv')
def test_cargar_datos_cluster_missing_columns(mock_read_csv):
    mock_read_csv.side_effect = [
        pd.DataFrame({'player_id': [1], 'name': ['Player A']}),  # Sin 'position'
        pd.DataFrame(MOCK_APPEARANCES_DATA)
    ]
    with pytest.raises(Exception) as exc_info:
        cargar_datos_cluster(tipo_posiciones=['Attack'], data_path='data')
    assert str(exc_info.value) == "Error al cargar datos: El archivo players.csv debe contener las columnas: player_id, name, position"

@patch('pandas.read_csv')
def test_cargar_datos_cluster_no_data(mock_read_csv):
    mock_read_csv.side_effect = [
        pd.DataFrame({'player_id': [1], 'name': ['Player A'], 'position': ['Defense']}),
        pd.DataFrame(MOCK_APPEARANCES_DATA)
    ]
    with pytest.raises(Exception) as exc_info:
        cargar_datos_cluster(tipo_posiciones=['Attack'], data_path='data')
    assert str(exc_info.value) == "Error al cargar datos: No se encontraron jugadores con posiciones Attack"

def test_generar_clusters_success():
    stats = pd.DataFrame({
        'goals': [2, 1, 0, 2, 1, 0, 2, 1, 0, 2],
        'assists': [1, 0, 2, 1, 0, 2, 1, 0, 2, 1],
        'minutes_played': [90, 80, 70, 90, 80, 70, 90, 80, 70, 90],
        'goals_per_game': [1.0, 0.5, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0, 1.0],
        'assists_per_game': [0.5, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5]
    })
    stats, _ = generar_clusters(stats)
    assert 'cluster' in stats.columns
    assert len(stats['cluster'].unique()) <= 10

def test_recomendar_success():
    stats = pd.DataFrame({
        'player_id': [1, 2, 3],
        'goals': [2, 1, 0],
        'assists': [1, 0, 2],
        'minutes_played': [90, 80, 70],
        'goals_per_game': [1.0, 0.5, 0.0],
        'assists_per_game': [0.5, 0.0, 1.0],
        'minutes_per_game': [90.0, 80.0, 70.0],
        'name': ['Player A', 'Player B', 'Player C'],
        'cluster': [0, 0, 1]
    }).set_index('player_id')
    with patch('utils.clustering.cargar_datos_cluster', return_value=stats), \
         patch('utils.clustering.generar_clusters', return_value=(stats, None)):
        all_stats, recommendations = recomendar(player_id_ref=1, tipo_posiciones=['Attack'], data_path='data')
        assert not recommendations.empty
        assert recommendations.iloc[0]['name'] == 'Player A'
        assert 'minutes_per_game' in recommendations.columns