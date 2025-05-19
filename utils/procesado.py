import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st

def plot_predicciones_arima(player_id, years_back=2, appearances_path="data/appearances.csv"):
    # Cargar datos
    try:
        df = pd.read_csv(appearances_path, parse_dates=["date"])
    except FileNotFoundError:
        st.error("No se encontró el archivo 'data/appearances.csv'.")
        return None

    # Filtrar por jugador
    player_df = df[df['player_id'] == player_id].copy()
    if player_df.empty:
        st.warning(f"No hay datos para el jugador con ID {player_id}.")
        return None

    # Obtener nombre del jugador (si players.csv existe)
    try:
        players_df = pd.read_csv("data/players.csv")
        player_name = players_df[players_df['player_id'] == player_id]['name'].iloc[0] if not players_df[players_df['player_id'] == player_id].empty else f"Jugador {player_id}"
    except FileNotFoundError:
        player_name = f"Jugador {player_id}"

    # Preprocesamiento
    player_df = player_df.sort_values(by='date')
    player_df.set_index('date', inplace=True)
    ts_df = player_df[['goals', 'assists']]

    # Reagrupar por mes
    monthly_goals = ts_df['goals'].resample('M').sum()
    monthly_assists = ts_df['assists'].resample('M').sum()

    # Filtrar datos históricos según years_back
    last_date = monthly_goals.index.max()
    start_date = last_date - pd.DateOffset(years=years_back)
    monthly_goals_recent = monthly_goals[monthly_goals.index >= start_date]
    monthly_assists_recent = monthly_assists[monthly_assists.index >= start_date]

    if monthly_goals_recent.empty or monthly_assists_recent.empty:
        st.warning(f"No hay datos suficientes en los últimos {years_back} años para el jugador {player_name}.")
        return None

    # ARIMA y predicción
    try:
        model_goals = ARIMA(monthly_goals, order=(10, 1, 10))
        result_goals = model_goals.fit()
        forecast_goals = result_goals.forecast(steps=12)
        future_dates = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=12, freq='M')
        forecast_goals.index = future_dates

        model_assists = ARIMA(monthly_assists, order=(10, 1, 10))
        result_assists = model_assists.fit()
        forecast_assists = result_assists.forecast(steps=12)
        forecast_assists.index = future_dates
    except Exception as e:
        st.error(f"Error al entrenar el modelo ARIMA: {e}")
        return None

    # Calcular estadísticas
    stats = {
        'total_goals': int(ts_df['goals'].sum()),
        'total_assists': int(ts_df['assists'].sum()),
        'matches_played': len(player_df),
        'avg_goals_per_month': round(monthly_goals.mean(), 2),
        'avg_assists_per_month': round(monthly_assists.mean(), 2)
    }

    # Gráfico de goles con Plotly
    fig_goals = go.Figure()
    fig_goals.add_trace(go.Scatter(
        x=monthly_goals_recent.index,
        y=monthly_goals_recent,
        mode='lines+markers',
        name='Histórico - Goles',
        line=dict(color='blue'),
        marker=dict(size=8)
    ))
    fig_goals.add_trace(go.Scatter(
        x=forecast_goals.index,
        y=forecast_goals,
        mode='lines',
        name='Predicción - Goles',
        line=dict(color='orange', dash='dash')
    ))
    fig_goals.add_vline(x=last_date, line=dict(color='gray', dash='dash'), name='Inicio predicción')
    fig_goals.update_layout(
        title=f'Goles Mensuales - {player_name}',
        xaxis_title='Fecha',
        yaxis_title='Goles por Mes',
        showlegend=True,
        template='plotly_white'
    )

    # Gráfico de asistencias con Plotly
    fig_assists = go.Figure()
    fig_assists.add_trace(go.Scatter(
        x=monthly_assists_recent.index,
        y=monthly_assists_recent,
        mode='lines+markers',
        name='Histórico - Asistencias',
        line=dict(color='green'),
        marker=dict(size=8)
    ))
    fig_assists.add_trace(go.Scatter(
        x=forecast_assists.index,
        y=forecast_assists,
        mode='lines',
        name='Predicción - Asistencias',
        line=dict(color='purple', dash='dash')
    ))
    fig_assists.add_vline(x=last_date, line=dict(color='gray', dash='dash'), name='Inicio predicción')
    fig_assists.update_layout(
        title=f'Asistencias Mensuales - {player_name}',
        xaxis_title='Fecha',
        yaxis_title='Asistencias por Mes',
        showlegend=True,
        template='plotly_white'
    )

    return player_name, stats, fig_goals, fig_assists