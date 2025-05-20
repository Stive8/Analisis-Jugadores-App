import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import euclidean_distances
from utils.procesado import plot_predicciones_arima
from utils.clustering import cargar_datos_cluster, generar_clusters, recomendar
from utils.minuto import sugerencias_por_intervalo
from utils.prediccion_resultado import entrenar_modelo, predecir_resultado, obtener_nombre_equipo, obtener_rolling_stats_equipo, obtener_url_escudo

# Set page configuration
st.set_page_config(page_title="Análisis de Jugadores", layout="wide")

# Load data and train model once
@st.cache_data
def load_data_and_model():
    clubs_df = pd.read_csv("data/clubs.csv")
    model, matches = entrenar_modelo()
    return clubs_df, model, matches

clubs_df, model, matches = load_data_and_model()

# Navigation
tabs = st.tabs([
    "📈 ARIMA - Series de Tiempo",
    "👥 Jugadores Similares",
    "⏱️ Sugerencias por Intervalo",
    "🔮 Predicción de Resultado"
])

# --- TAB 1: ARIMA ---
with tabs[0]:
    st.title("📈 Predicción de Rendimiento con ARIMA")
    st.markdown("""
    Este módulo utiliza el modelo ARIMA para predecir los **goles** y **asistencias** mensuales de un jugador en los próximos 12 meses, basado en su rendimiento histórico.  
    Selecciona un jugador y un período de tiempo para visualizar los datos históricos y las predicciones.
    """)

    # Cargar lista de jugadores (asumiendo que existe players.csv)
    try:
        players_df = pd.read_csv("data/players.csv")
        player_options = {row['name']: row['player_id'] for _, row in players_df.iterrows()}
    except FileNotFoundError:
        player_options = None
        st.warning("No se encontró 'data/players.csv'. Usa el ID del jugador manualmente.")

    # Selección de jugador
    col1, col2 = st.columns([2, 1])
    with col1:
        if player_options:
            player_name = st.selectbox("Selecciona un jugador:", options=sorted(player_options.keys()), key="arima_player")
            player_id = player_options[player_name]
        else:
            player_id = st.text_input("Ingresa el ID del jugador:", "", key="arima_player_id")
    with col2:
        years_back = st.slider("Años de datos históricos:", min_value=1, max_value=5, value=2, key="arima_years")

    if (player_id.isdigit() if not player_options else player_name):
        try:
            # Llamar a la función ARIMA con el número de años
            result = plot_predicciones_arima(
                int(player_id) if not player_options else player_id,
                years_back=years_back
            )
            if result:
                player_name, stats, fig_goals, fig_assists = result
                # Mostrar estadísticas del jugador
                st.subheader(f"Estadísticas de {player_name}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Goles", f"{stats['total_goals']}")
                col2.metric("Total Asistencias", f"{stats['total_assists']}")
                col3.metric("Partidos Jugados", f"{stats['matches_played']}")

                # Mostrar gráficos
                st.subheader("Predicción de Goles")
                st.plotly_chart(fig_goals, use_container_width=True)
                st.subheader("Predicción de Asistencias")
                st.plotly_chart(fig_assists, use_container_width=True)
        except Exception as e:
            st.error(f"Error al generar predicciones: {e}")
    elif player_id and not player_id.isdigit():
        st.error("Por favor, ingresa un ID numérico válido.")

# --- TAB 2: Clustering ---
with tabs[1]:
    st.title("👥 Jugadores Similares")
    st.markdown("""
    Este módulo utiliza clustering para identificar jugadores con un rendimiento similar al seleccionado, 
    basado en estadísticas como goles, asistencias y minutos jugados.  
    Selecciona un jugador y su posición para ver una lista de los 10 jugadores más cercanos en el mismo clúster.
    """)

    # Cargar lista de jugadores
    player_options = None
    try:
        players_df = pd.read_csv("data/players.csv")
        if not all(col in players_df.columns for col in ['player_id', 'name']):
            raise ValueError("El archivo players.csv debe contener las columnas: player_id, name")
        player_options = {row['name']: row['player_id'] for _, row in players_df.iterrows()}
    except FileNotFoundError:
        st.error("No se encontró 'data/players.csv'. Verifica la ruta del archivo.")
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Error al cargar players.csv: {str(e)}")

    if player_options:
        player_name = st.selectbox(
            "Selecciona un jugador:",
            options=sorted(player_options.keys()),
            key="rec_player"
        )
        player_id = player_options[player_name]
    else:
        player_id = st.text_input("Ingresa el ID del jugador:", "", key="rec_id")

    # Selección de posiciones con dos checkboxes
    st.markdown("**Selecciona las posiciones para buscar:**")
    col1, col2 = st.columns(2)
    with col1:
        buscar_atacantes = st.checkbox("Atacantes", value=True, key="pos_attack")
    with col2:
        buscar_centrocampistas = st.checkbox("Centrocampistas", value=False, key="pos_midfield")

    # Determinar las posiciones seleccionadas
    tipo_posiciones = []
    if buscar_atacantes:
        tipo_posiciones.append("Attack")
    if buscar_centrocampistas:
        tipo_posiciones.append("Midfield")

    # Validar que se haya seleccionado al menos una posición
    if not tipo_posiciones:
        st.warning("Por favor, selecciona al menos una posición (Atacantes o Centrocampistas).")
        tipo_posiciones = ["Attack"]  # Valor por defecto si no se selecciona ninguna

    if player_id:
        with st.spinner("Procesando datos de clustering..."):
            try:
                # Validar que player_id sea numérico
                if not player_options and not player_id.isdigit():
                    raise ValueError("El ID del jugador debe ser numérico")
                player_id = int(player_id)

                df_stats, recomendaciones = recomendar(player_id, tipo_posiciones)
                if recomendaciones is not None and not recomendaciones.empty:
                    # Mostrar estadísticas del jugador seleccionado
                    display_name = player_name if player_options else f"ID {player_id}"
                    posiciones_str = ", ".join(tipo_posiciones)  # Convertir lista de posiciones a string
                    st.subheader(f"Estadísticas de {display_name} ({posiciones_str})")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Goles", f"{int(df_stats.loc[player_id, 'goals'])}")
                    col2.metric("Total Asistencias", f"{int(df_stats.loc[player_id, 'assists'])}")
                    col3.metric("Partidos Jugados", f"{int(df_stats.loc[player_id, 'appearances'])}")

                    # Calcular distancia euclidiana para encontrar los 10 más cercanos
                    features = ['goals', 'assists', 'minutes_played', 'goals_per_game', 'assists_per_game']
                    if not all(col in recomendaciones.columns for col in features):
                        st.error(f"Faltan columnas en recomendaciones: {', '.join(features)}")
                    else:
                        # Extraer características del jugador seleccionado
                        player_features = df_stats.loc[player_id, features].values.reshape(1, -1)
                        # Extraer características de los jugadores recomendados
                        recomendaciones_features = recomendaciones[features].values
                        # Calcular distancias euclidianas
                        distances = euclidean_distances(player_features, recomendaciones_features).flatten()
                        # Añadir distancias al DataFrame
                        recomendaciones['distance'] = distances
                        # Excluir al jugador seleccionado (si está en recomendaciones)
                        recomendaciones = recomendaciones[recomendaciones.index != player_id]
                        # Ordenar por distancia y tomar los 10 más cercanos
                        top_10_recomendaciones = recomendaciones.sort_values(by='distance').head(10)

                        # Mostrar tabla de los 10 jugadores más cercanos
                        st.subheader(f"Los 10 jugadores más similares a {display_name}")
                        formatted_recommendations = top_10_recomendaciones.copy()
                        formatted_recommendations = formatted_recommendations.rename(columns={
                            'name': 'Nombre',
                            'goals': 'Goles',
                            'assists': 'Asistencias',
                            'minutes_played': 'Minutos Jugados',
                            'goals_per_game': 'Goles por Partido',
                            'assists_per_game': 'Asistencias por Partido',
                            'minutes_per_game': 'Minutos por Partido'
                        })
                        formatted_recommendations['Goles por Partido'] = formatted_recommendations['Goles por Partido'].round(2)
                        formatted_recommendations['Asistencias por Partido'] = formatted_recommendations['Asistencias por Partido'].round(2)
                        formatted_recommendations['Minutos por Partido'] = formatted_recommendations['Minutos por Partido'].round(2)
                        # Excluir la columna 'distance' de la tabla
                        formatted_recommendations = formatted_recommendations.drop(columns=['distance'])
                        st.dataframe(
                            formatted_recommendations,
                            use_container_width=True,
                            hide_index=True
                        )

                        # Visualización con Plotly
                        try:
                            required_cols = ['goals_per_game', 'assists_per_game', 'name']
                            if all(col in top_10_recomendaciones.columns for col in required_cols) and all(col in df_stats.columns for col in required_cols):
                                # Filtrar filas con valores no nulos
                                top_10_recomendaciones = top_10_recomendaciones.dropna(subset=['goals_per_game', 'assists_per_game'])
                                if not top_10_recomendaciones.empty:
                                    st.subheader("Visualización de Jugadores Similares")
                                    st.markdown("El gráfico muestra los 10 jugadores más cercanos en el mismo clúster, comparando goles y asistencias por partido.")

                                    # Crear DataFrame para Plotly
                                    plot_data = top_10_recomendaciones[['name', 'goals_per_game', 'assists_per_game']].copy()
                                    plot_data['Type'] = 'Jugadores Similares'

                                    # Datos del jugador seleccionado
                                    if pd.notna(df_stats.loc[player_id, 'goals_per_game']) and pd.notna(df_stats.loc[player_id, 'assists_per_game']):
                                        selected_player = pd.DataFrame({
                                            'name': [df_stats.loc[player_id, 'name']],
                                            'goals_per_game': [df_stats.loc[player_id, 'goals_per_game']],
                                            'assists_per_game': [df_stats.loc[player_id, 'assists_per_game']],
                                            'Type': [f"{display_name} (Seleccionado)"]
                                        })
                                        plot_data = pd.concat([plot_data, selected_player], ignore_index=True)
                                    else:
                                        st.warning("El jugador seleccionado no tiene datos suficientes para graficar (goals_per_game o assists_per_game faltantes).")

                                    # Crear gráfico si hay datos
                                    if not plot_data.empty:
                                        fig = px.scatter(
                                            plot_data,
                                            x='goals_per_game',
                                            y='assists_per_game',
                                            color='Type',
                                            hover_data=['name'],
                                            labels={'goals_per_game': 'Goles por Partido', 'assists_per_game': 'Asistencias por Partido'},
                                            title='Jugadores Similares (Top 10)'
                                        )
                                        fig.update_traces(marker=dict(size=10), selector=dict(name='Jugadores Similares'))
                                        fig.update_traces(marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')), selector=dict(name=f"{display_name} (Seleccionado)"))
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("No hay datos suficientes para generar el gráfico.")
                                else:
                                    st.warning("No se pueden generar gráficos: los jugadores recomendados tienen datos faltantes (goals_per_game o assists_per_game).")
                            else:
                                st.warning("No se pueden generar gráficos debido a columnas faltantes (goals_per_game, assists_per_game o name).")
                        except Exception as e:
                            st.error(f"Error al generar el gráfico: {str(e)}")

                        # Opción para descargar la tabla
                        csv = formatted_recommendations.to_csv(index=False)
                        st.download_button(
                            label="Descargar tabla como CSV",
                            data=csv,
                            file_name=f"jugadores_similares_{display_name}.csv",
                            mime="text/csv"
                        )

                else:
                    st.warning("Jugador no encontrado o sin datos suficientes.")
            except ValueError as e:
                st.error(f"Error de datos: {str(e)}")
            except KeyError as e:
                st.error(f"Error: No se encontró la columna o índice {str(e)} en los datos")
            except Exception as e:
                st.error(f"Error en clustering: {str(e)}")
    else:
        st.info("Por favor, selecciona un jugador para ver las recomendaciones.")

# --- TAB 3: Interval Suggestions ---
with tabs[2]:
    st.title("⏱️ Sugerencias por Intervalo de Tiempo")
    intervalo = st.text_input("Intervalo de minutos (ej. 61-70):", "61-70")
    if intervalo:
        try:
            df_intervalo, X_scaled, kmeans_model = sugerencias_por_intervalo(intervalo)
            if df_intervalo is None or df_intervalo.empty:
                st.warning("No se encontraron datos para ese intervalo.")
            else:
                st.write("Top jugadores por goles y tarjetas:")
                st.dataframe(df_intervalo.sort_values(by=['goals', 'cards'], ascending=False)[
                    ['player_name', 'position', 'goals', 'cards', 'cluster']
                ].head(10))

                # Plotly scatter plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=X_scaled[:, 0], y=X_scaled[:, 1],
                    mode='markers',
                    marker=dict(color=df_intervalo['cluster'], colorscale='Viridis', size=10, line=dict(width=1, color='black')),
                    text=df_intervalo['player_name'],
                    hoverinfo='text'
                ))
                fig.update_layout(
                    title=f"Clusters de Jugadores ({intervalo})",
                    xaxis_title="Goles (escalados)",
                    yaxis_title="Tarjetas (escaladas)",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error en sugerencias por intervalo: {e}")

# --- TAB 4: Match Prediction ---
with tabs[3]:
    st.title("🔮 Predicción de Resultado entre Equipos")
    st.markdown("""
    Este módulo predice el resultado probable de un partido entre dos equipos basándose en sus estadísticas recientes (últimos 5 partidos).  
    Selecciona los equipos local y visitante, luego haz clic en **Predecir** para ver el resultado y las estadísticas detalladas.
    """)

    # Cargar lista de equipos desde clubs.csv
    try:
        clubs_df = pd.read_csv("data/clubs.csv")
        team_options = {row['name']: row['club_id'] for _, row in clubs_df.iterrows()}
    except FileNotFoundError:
        team_options = None
        st.error("No se encontró 'data/clubs.csv'. Usa IDs de equipos manualmente.")

    # Selección de equipos
    col1, col2 = st.columns(2)
    with col1:
        if team_options:
            home_team_name = st.selectbox("Selecciona el equipo local:", options=sorted(team_options.keys()), key="home_team")
            home_id = team_options[home_team_name]
        else:
            home_id = st.number_input("ID del equipo local", min_value=0, step=1, key="home_team_id")
            home_team_name = obtener_nombre_equipo(home_id, clubs_df) if home_id > 0 else "Equipo no seleccionado"
    with col2:
        if team_options:
            away_team_name = st.selectbox("Selecciona el equipo visitante:", options=sorted(team_options.keys()), key="away_team")
            away_id = team_options[away_team_name]
        else:
            away_id = st.number_input("ID del equipo visitante", min_value=0, step=1, key="away_team_id")
            away_team_name = obtener_nombre_equipo(away_id, clubs_df) if away_id > 0 else "Equipo no seleccionado"

    # Botón para predecir
    if st.button("Predecir Resultado", key="predict_button"):
        if home_id == away_id:
            st.warning("Los equipos no pueden ser iguales. Por favor, selecciona equipos diferentes.")
        elif home_id == 0 or away_id == 0:
            st.error("Por favor, selecciona ambos equipos (IDs mayores a 0).")
        else:
            try:
                # Obtener datos de los equipos
                stats_home = obtener_rolling_stats_equipo(matches, home_id)
                stats_away = obtener_rolling_stats_equipo(matches, away_id)

                if stats_home.empty or stats_away.empty:
                    st.error(f"No hay datos suficientes para {home_team_name} o {away_team_name}.")
                else:
                    # Estadísticas resumidas
                    last_home_stats = stats_home.iloc[-1]
                    last_away_stats = stats_away.iloc[-1]
                    st.subheader("Estadísticas Recientes")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**{home_team_name} (Local)**")
                        # Obtener la URL del escudo para el equipo local
                        url_home = clubs_df.loc[clubs_df["club_id"] == home_id, "url"].values[0] if home_id in clubs_df["club_id"].values else None
                        if url_home:
                            img_url_home = obtener_url_escudo(url_home)
                            if img_url_home:
                                st.image(img_url_home, width=100)  # Reducir a 100 píxeles
                            else:
                                st.warning(f"No se pudo cargar el escudo de {home_team_name}.")
                        else:
                            st.warning(f"No se encontró URL para {home_team_name}.")
                        st.metric("Promedio Goles a Favor", f"{last_home_stats['goals_for_rolling']:.2f}")
                        st.metric("Promedio Goles en Contra", f"{last_home_stats['goals_against_rolling']:.2f}")
                        st.metric("Tasa de Victoria", f"{last_home_stats['win_rate_rolling']:.2f}")
                    with col2:
                        st.markdown(f"**{away_team_name} (Visitante)**")
                        # Obtener la URL del escudo para el equipo visitante
                        url_away = clubs_df.loc[clubs_df["club_id"] == away_id, "url"].values[0] if away_id in clubs_df["club_id"].values else None
                        if url_away:
                            img_url_away = obtener_url_escudo(url_away)
                            if img_url_away:
                                st.image(img_url_away, width=100)  # Reducir a 100 píxeles
                            else:
                                st.warning(f"No se pudo cargar el escudo de {away_team_name}.")
                        else:
                            st.warning(f"No se encontró URL para {away_team_name}.")
                        st.metric("Promedio Goles a Favor", f"{last_away_stats['goals_for_rolling']:.2f}")
                        st.metric("Promedio Goles en Contra", f"{last_away_stats['goals_against_rolling']:.2f}")
                        st.metric("Tasa de Victoria", f"{last_away_stats['win_rate_rolling']:.2f}")

                    # Predicción
                    resultado, prob = predecir_resultado(model, matches, home_id, away_id)
                    st.markdown(f"### 🧠 Predicción: **{resultado}**")
                    st.write(f"**Probabilidades:** Local {round(prob[2]*100, 1)}% | Empate {round(prob[1]*100, 1)}% | Visitante {round(prob[0]*100, 1)}%")

                    # Gráficos de estadísticas rolling
                    with st.expander("📊 Ver estadísticas detalladas de ambos equipos", expanded=False):
                        # Gráfico para equipo local
                        st.subheader(f"📈 {home_team_name}")
                        fig_home = go.Figure()
                        fig_home.add_trace(go.Scatter(x=stats_home['date'], y=stats_home['goals_for_rolling'], mode='lines+markers', name='Goles a Favor', line=dict(color='blue')))
                        fig_home.add_trace(go.Scatter(x=stats_home['date'], y=stats_home['goals_against_rolling'], mode='lines+markers', name='Goles en Contra', line=dict(color='red')))
                        fig_home.add_trace(go.Scatter(x=stats_home['date'], y=stats_home['goal_diff_rolling'], mode='lines+markers', name='Diferencia de Goles', line=dict(color='green')))
                        fig_home.add_trace(go.Scatter(x=stats_home['date'], y=stats_home['win_rate_rolling'], mode='lines+markers', name='Tasa de Victoria', line=dict(color='purple')))
                        fig_home.update_layout(
                            title=f'Estadísticas Rolling - {home_team_name}',
                            xaxis_title='Fecha',
                            yaxis_title='Valor',
                            template='plotly_white',
                            showlegend=True
                        )
                        st.plotly_chart(fig_home, use_container_width=True)

                        # Gráfico para equipo visitante
                        st.subheader(f"📉 {away_team_name}")
                        fig_away = go.Figure()
                        fig_away.add_trace(go.Scatter(x=stats_away['date'], y=stats_away['goals_for_rolling'], mode='lines+markers', name='Goles a Favor', line=dict(color='blue')))
                        fig_away.add_trace(go.Scatter(x=stats_away['date'], y=stats_away['goals_against_rolling'], mode='lines+markers', name='Goles en Contra', line=dict(color='red')))
                        fig_away.add_trace(go.Scatter(x=stats_away['date'], y=stats_away['goal_diff_rolling'], mode='lines+markers', name='Diferencia de Goles', line=dict(color='green')))
                        fig_away.add_trace(go.Scatter(x=stats_away['date'], y=stats_away['win_rate_rolling'], mode='lines+markers', name='Tasa de Victoria', line=dict(color='purple')))
                        fig_away.update_layout(
                            title=f'Estadísticas Rolling - {away_team_name}',
                            xaxis_title='Fecha',
                            yaxis_title='Valor',
                            template='plotly_white',
                            showlegend=True
                        )
                        st.plotly_chart(fig_away, use_container_width=True)
            except Exception as e:
                st.error(f"Error al generar la predicción: {e}")