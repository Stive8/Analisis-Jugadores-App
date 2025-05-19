import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.procesado import plot_predicciones_arima
from utils.clustering import cargar_datos_cluster, generar_clusters, recomendar
from utils.minuto import sugerencias_por_intervalo
from utils.prediccion_resultado import entrenar_modelo, predecir_resultado, obtener_nombre_equipo, obtener_rolling_stats_equipo

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
    player_id = st.text_input("ID del jugador para recomendaciones:", key="rec_id")
    tipo = st.checkbox("¿Buscar en centrocampistas?", key="pos_toggle")
    tipo_pos = "Midfield" if tipo else "Attack"
    if player_id.isdigit():
        try:
            df_stats, recomendaciones = recomendar(int(player_id), tipo_pos)
            if recomendaciones is not None:
                st.write("Jugadores similares:")
                st.dataframe(recomendaciones)
            else:
                st.warning("Jugador no encontrado o sin datos suficientes.")
        except Exception as e:
            st.error(f"Error en clustering: {e}")

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
                        url_home = clubs_df.loc[clubs_df["club_id"] == home_id, "url"].values[0] if home_id in clubs_df["club_id"].values else None
                        if url_home:
                            st.image(url_home, width=150)
                        st.metric("Promedio Goles a Favor", f"{last_home_stats['goals_for_rolling']:.2f}")
                        st.metric("Promedio Goles en Contra", f"{last_home_stats['goals_against_rolling']:.2f}")
                        st.metric("Tasa de Victoria", f"{last_home_stats['win_rate_rolling']:.2f}")
                    with col2:
                        st.markdown(f"**{away_team_name} (Visitante)**")
                        url_away = clubs_df.loc[clubs_df["club_id"] == away_id, "url"].values[0] if away_id in clubs_df["club_id"].values else None
                        if url_away:
                            st.image(url_away, width=150)
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