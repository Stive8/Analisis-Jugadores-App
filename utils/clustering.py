import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cargar_datos_cluster(tipo_posiciones=['Attack'], data_path='data'):
    try:
        # Cargar archivos
        df_players = pd.read_csv(f"{data_path}/players.csv")
        df_appearances = pd.read_csv(f"{data_path}/appearances.csv", parse_dates=["date"])

        # Validar columnas requeridas
        required_players = ['player_id', 'name', 'position']
        required_appearances = ['player_id', 'player_name', 'goals', 'assists', 'minutes_played', 'game_id', 'date']
        if not all(col in df_players.columns for col in required_players):
            raise ValueError(f"El archivo players.csv debe contener las columnas: {', '.join(required_players)}")
        if not all(col in df_appearances.columns for col in required_appearances):
            raise ValueError(f"El archivo appearances.csv debe contener las columnas: {', '.join(required_appearances)}")

        # Filtrar jugadores por las posiciones seleccionadas
        jugadores = df_players[df_players['position'].str.contains('|'.join(tipo_posiciones), case=False, na=False)]
        if jugadores.empty:
            raise ValueError(f"No se encontraron jugadores con posiciones {', '.join(tipo_posiciones)}")
        df_apariciones = df_appearances[df_appearances['player_id'].isin(jugadores['player_id'])]

        if df_apariciones.empty:
            raise ValueError(f"No hay datos de apariciones para jugadores con posiciones {', '.join(tipo_posiciones)}")

        # Agregación: Contar apariciones únicas por player_id basado en game_id
        stats = df_apariciones.groupby('player_id').agg({
            'player_name': 'first',
            'goals': 'sum',
            'assists': 'sum',
            'minutes_played': 'sum',
            'game_id': 'nunique'  # Contar juegos únicos
        }).rename(columns={'game_id': 'appearances'})

        # Evitar división por cero
        stats['goals_per_game'] = stats['goals'] / stats['appearances'].replace(0, 1)
        stats['assists_per_game'] = stats['assists'] / stats['appearances'].replace(0, 1)
        stats['minutes_per_game'] = stats['minutes_played'] / stats['appearances'].replace(0, 1)

        # Añadir nombre desde players.csv
        stats = stats.reset_index().merge(
            df_players[['player_id', 'name']], on='player_id', how='left'
        ).set_index('player_id')

        # Manejar nombres nulos
        if stats['name'].isna().any():
            stats['name'] = stats['name'].fillna(stats['player_name'])

        return stats
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontraron los archivos en {data_path}. Verifica 'players.csv' y 'appearances.csv'.")
    except Exception as e:
        raise Exception(f"Error al cargar datos: {str(e)}")

def generar_clusters(stats):
    try:
        features = ['goals', 'assists', 'minutes_played', 'goals_per_game', 'assists_per_game']
        if not all(col in stats.columns for col in features):
            raise ValueError(f"Faltan columnas en los datos: {', '.join(features)}")
        X = stats[features].fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=10, random_state=42)
        stats['cluster'] = kmeans.fit_predict(X_scaled)

        return stats, kmeans
    except Exception as e:
        raise Exception(f"Error al generar clusters: {str(e)}")

def recomendar(player_id_ref, tipo_posiciones=['Attack'], data_path='data'):
    try:
        stats = cargar_datos_cluster(tipo_posiciones, data_path)
        stats, _ = generar_clusters(stats)

        if player_id_ref not in stats.index:
            return None, None

        cluster_objetivo = stats.loc[player_id_ref, 'cluster']
        recomendaciones = stats[stats['cluster'] == cluster_objetivo].sort_values(by='goals', ascending=False)

        # Incluir todas las columnas relevantes
        return stats, recomendaciones[['name', 'goals', 'assists', 'minutes_played', 'goals_per_game', 'assists_per_game', 'minutes_per_game']]
    except Exception as e:
        raise Exception(f"Error en la recomendación: {str(e)}")
