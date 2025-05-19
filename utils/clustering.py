import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cargar_datos_cluster(tipo_posicion='Attack', data_path='data'):
    # Cargar archivos
    df_players = pd.read_csv(f"{data_path}/players.csv")
    df_appearances = pd.read_csv(f"{data_path}/appearances.csv", parse_dates=["date"])

    # Filtrar jugadores por posición
    jugadores = df_players[df_players['position'] == tipo_posicion]
    df_apariciones = df_appearances[df_appearances['player_id'].isin(jugadores['player_id'])]

    # Agregación
    stats = df_apariciones.groupby('player_id').agg({
        'player_name': 'first',
        'goals': 'sum',
        'assists': 'sum',
        'minutes_played': 'sum',
        'game_id': 'count'
    }).rename(columns={'game_id': 'appearances'})

    stats['goals_per_game'] = stats['goals'] / stats['appearances']
    stats['assists_per_game'] = stats['assists'] / stats['appearances']
    stats['minutes_per_game'] = stats['minutes_played'] / stats['appearances']

    # Añadir nombre desde players.csv por seguridad
    stats = stats.reset_index().merge(
        jugadores[['player_id', 'name']], on='player_id', how='left'
    ).set_index('player_id')

    return stats

def generar_clusters(stats):
    features = ['goals', 'assists', 'minutes_played', 'goals_per_game', 'assists_per_game']
    X = stats[features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=10, random_state=42)
    stats['cluster'] = kmeans.fit_predict(X_scaled)

    return stats, kmeans

def recomendar(player_id_ref, tipo_posicion='Attack', data_path='data'):
    stats = cargar_datos_cluster(tipo_posicion, data_path)
    stats, _ = generar_clusters(stats)

    if player_id_ref not in stats.index:
        return None, None

    cluster_objetivo = stats.loc[player_id_ref, 'cluster']
    recomendaciones = stats[stats['cluster'] == cluster_objetivo].sort_values(by='goals', ascending=False)

    return stats, recomendaciones[['name', 'goals', 'assists', 'minutes_played', 'goals_per_game']]
