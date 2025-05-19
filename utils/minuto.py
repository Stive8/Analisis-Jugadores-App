# utils/minuto.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def sugerencias_por_intervalo(intervalo: str, data_path="data"):
    # Cargar datos
    players = pd.read_csv(f"{data_path}/players.csv")
    events = pd.read_csv(f"{data_path}/game_events.csv")

    # Filtrar eventos relevantes
    filtered_events = events[events['type'].isin(['Goals', 'Cards'])].copy()
    filtered_events['minute'] = pd.to_numeric(filtered_events['minute'], errors='coerce')
    filtered_events.dropna(subset=['minute'], inplace=True)

    # Crear intervalos
    def crear_intervalo(minuto, tam_intervalo=10):
        start = (minuto // tam_intervalo) * tam_intervalo + 1
        end = start + tam_intervalo - 1
        return f"{int(start)}-{int(end)}"

    filtered_events['minute_interval'] = filtered_events['minute'].apply(crear_intervalo)

    # Agrupar eventos
    goals = filtered_events[filtered_events['type'] == 'Goals'].groupby(
        ['player_id', 'minute_interval']).size().reset_index(name='goals')
    cards = filtered_events[filtered_events['type'] == 'Cards'].groupby(
        ['player_id', 'minute_interval']).size().reset_index(name='cards')

    # Unir goles y tarjetas
    df_minute_stats = pd.merge(goals, cards, how='outer', on=['player_id', 'minute_interval']).fillna(0)
    df_minute_stats[['goals', 'cards']] = df_minute_stats[['goals', 'cards']].astype(int)

    # Agregar info del jugador
    df_minute_stats = df_minute_stats.merge(players[['player_id', 'name', 'position']], on='player_id', how='left')
    df_minute_stats.rename(columns={'name': 'player_name'}, inplace=True)

    # Filtrar por intervalo
    stats_interval = df_minute_stats[df_minute_stats['minute_interval'] == intervalo].copy()

    if stats_interval.empty:
        return None, None, None

    # Clustering
    features = ['goals', 'cards']
    X = stats_interval[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    stats_interval['cluster'] = kmeans.fit_predict(X_scaled)

    return stats_interval, X_scaled, kmeans
