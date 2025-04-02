import base64
import io
import logging
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash import dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import datetime
import chardet
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import zipfile
from sklearn.metrics import mean_absolute_error, mean_squared_error
import functools
import warnings
try:
    import pdfkit
except ImportError:
    pdfkit = None
import plotly.io as pio

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True, title="📊 Extraordinary Data Analysis Dashboard")
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .dropdown-purple .Select-control,
            .dropdown-purple .Select-menu-outer {
                background-color: #2e3b55 !important;
                color: black !important;
            }
            .dropdown-purple .Select-value-label,
            .dropdown-purple .Select-placeholder,
            .dropdown-purple .Select-option,
            .dropdown-purple .Select-single-value {
                color: black !important;
            }
            .dropdown-purple .Select-option.is-focused {
                background-color: #1f2a44 !important;
                color: white !important;
            }
            .kpi-card {
                height: 120px !important;
                background-color: #2e3b55 !important;
                color: white !important;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
            }
            .kpi-card h6 {
                font-size: 1rem !important;
                margin-bottom: 0.5rem !important;
            }
            .kpi-card h4 {
                font-size: 1.5rem !important;
                color: #00d4ff !important;
            }
            .kpi-card p {
                font-size: 0.9rem !important;
                margin-bottom: 0 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

class DataStore:
    def __init__(self):
        self.df = None
        self.last_updated = None
        self.filtered_df = None
        self.raw_df = None
        self.original_row_count = None
        self.comments = []

data_store = DataStore()

def cache(func):
    cache_dict = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key in cache_dict:
            return cache_dict[key]
        result = func(*args, **kwargs)
        cache_dict[key] = result
        return result
    return wrapper

def clean_and_validate_data(df):
    try:
        if data_store.raw_df is None:
            data_store.raw_df = df.copy()
            data_store.original_row_count = len(df)
        simplified_messages = []
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.lower().str.strip()
        numerical_cols = [col for col in df.columns if any(k in col.lower() for k in ['price', 'sales', 'revenue', 'stock', 'quantity', 'amount'])]
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(how='all', axis=1)
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        df = df.drop_duplicates(keep='first')
        return df, simplified_messages
    except Exception as e:
        logger.error(f"Data cleaning error: {e}")
        return None, [f"Error during cleaning: {str(e)}"]

def parse_contents(contents, filename, max_rows=5000):
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        encoding = chardet.detect(decoded)['encoding'] or 'utf-8'
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode(encoding)), low_memory=False, engine='c')
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')
        else:
            raise ValueError("Unsupported file format")
        if df.empty:
            return None, ["Uploaded file is empty"]
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42)
            simplified_messages = [f"Dataset downsampled to {max_rows} rows"]
        else:
            simplified_messages = []
        df, cleaning_messages = clean_and_validate_data(df)
        if df is None:
            return None, cleaning_messages
        simplified_messages.extend(cleaning_messages)
        return df, simplified_messages
    except Exception as e:
        logger.error(f"File parsing error: {e}")
        return None, [f"Error parsing file: {str(e)}"]

@cache
def generate_main_chart(df, x_axis, y_axes, chart_type, dark_mode, selected_category):
    try:
        logger.info(f"Generating main chart with x_axis: {x_axis}, y_axes: {y_axes}, chart_type: {chart_type}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Unique values in {x_axis}: {df[x_axis].unique() if x_axis in df.columns else 'N/A'}")
        
        if not x_axis or not y_axes or x_axis not in df.columns or not all(y in df.columns for y in y_axes):
            logger.error("Invalid X or Y axis selected")
            return px.scatter(title="Please select valid X and Y axes")

        # Validate Y-axes are numerical for bar chart
        if chart_type == 'bar' and not all(pd.api.types.is_numeric_dtype(df[y]) for y in y_axes):
            logger.error(f"One or more Y-axes {y_axes} are not numerical for bar chart")
            return px.scatter(title=f"Y-axes must be numerical for bar chart")

        # Sample data only if necessary
        if len(df) > 500:
            df = df.sample(500, random_state=42)
            logger.info("Data sampled to 500 rows")

        # Ensure sampled data is not empty
        if df.empty:
            logger.error("Sampled data is empty")
            return px.scatter(title="Sampled data is empty")

        template = 'plotly_dark' if dark_mode else 'plotly'

        if chart_type == 'bar':
            grouped_df = df.groupby(x_axis)[y_axes].sum().reset_index()
            logger.info(f"Grouped DataFrame for bar chart:\n{grouped_df}")
            fig = go.Figure()
            for y_axis in y_axes:
                fig.add_trace(go.Bar(x=grouped_df[x_axis], y=grouped_df[y_axis], name=y_axis))
            fig.update_layout(barmode='group', template=template, title="Bar Chart Comparison")
            if selected_category:
                for trace in fig.data:
                    trace.marker.color = ['#00d4ff' if x == selected_category else '#636efa' for x in grouped_df[x_axis]]

        elif chart_type == 'pie':
            if len(y_axes) > 1:
                return px.scatter(title="Pie chart supports only one Y-axis")
            fig = px.pie(df, names=x_axis, values=y_axes[0], template=template)
            if selected_category:
                explode = [0.1 if x == selected_category else 0 for x in df[x_axis].unique()]
                fig.update_traces(pull=explode, marker=dict(colors=['#00d4ff' if x == selected_category else '#636efa' for x in df[x_axis].unique()]))

        elif chart_type == 'scatter':
            fig = go.Figure()
            for y_axis in y_axes:
                fig.add_trace(go.Scatter(x=df[x_axis], y=df[y_axis], mode='markers', name=y_axis))
            fig.update_layout(template=template, title="Scatter Plot Comparison")
            if selected_category:
                for trace in fig.data:
                    trace.marker.size = [15 if x == selected_category else 8 for x in df[x_axis]]

        elif chart_type == 'line':
            fig = go.Figure()
            for y_axis in y_axes:
                fig.add_trace(go.Scatter(x=df[x_axis], y=df[y_axis], mode='lines', name=y_axis))
            fig.update_layout(template=template, title="Line Chart Comparison")
            if selected_category:
                for y_axis in y_axes:
                    fig.add_scatter(x=[selected_category], y=[df[df[x_axis] == selected_category][y_axis].iloc[0]], mode='markers', marker=dict(size=15, color='#00d4ff'), showlegend=False)

        elif chart_type == 'heatmap':
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numerical_cols) >= 2:
                corr_matrix = df[numerical_cols].corr()
                fig = ff.create_annotated_heatmap(z=corr_matrix.values, x=numerical_cols, y=numerical_cols, colorscale='Plasma' if dark_mode else 'Viridis')
            else:
                fig = px.scatter(title="Not enough numerical columns for heatmap")

        elif chart_type == 'box':
            fig = go.Figure()
            for y_axis in y_axes:
                fig.add_trace(go.Box(x=df[x_axis], y=df[y_axis], name=y_axis))
            fig.update_layout(template=template, title="Box Plot Comparison")

        elif chart_type == 'regression':
            if len(y_axes) > 1:
                return px.scatter(title="Regression supports only one Y-axis")
            y_axis = y_axes[0]
            if (df[x_axis].dtype in ['int64', 'float64'] and df[y_axis].dtype in ['int64', 'float64']):
                X = df[[x_axis]].dropna()
                y = df.loc[X.index, y_axis]
                model = LinearRegression()
                model.fit(X, y)
                df.loc[X.index, 'predicted'] = model.predict(X)
                fig = px.scatter(df, x=x_axis, y=y_axis, trendline="ols", template=template)
                if selected_category:
                    fig.update_traces(marker=dict(size=[15 if x == selected_category else 8 for x in df[x_axis]]))
            else:
                fig = px.scatter(title="Regression requires numerical X and Y axes")

        else:
            fig = px.scatter(title="Unsupported chart type")

        logger.info("Main chart generated successfully")
        return fig
    except Exception as e:
        logger.error(f"Main chart error: {e}")
        return px.scatter(title=f"Error generating main chart: {str(e)}")

@cache
def generate_pie_chart(df, names_col, dark_mode, selected_category):
    try:
        if not names_col or names_col not in df.columns:
            return px.scatter(title="Please select a column for the Pie Chart")
        
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numerical_cols:
            return px.scatter(title="No numerical columns available for Pie Chart values")
        values_col = numerical_cols[0]

        if len(df) > 500:
            df = df.sample(500, random_state=42)
        
        fig = px.pie(df, names=names_col, values=values_col, template='plotly_dark' if dark_mode else 'plotly', title=f"Pie: {names_col} Distribution")
        if selected_category:
            explode = [0.1 if x == selected_category else 0 for x in df[names_col].unique()]
            fig.update_traces(pull=explode, marker=dict(colors=['#00d4ff' if x == selected_category else '#636efa' for x in df[names_col].unique()]))
        return fig
    except Exception as e:
        logger.error(f"Pie chart error: {e}")
        return px.scatter(title=f"Error generating pie chart: {str(e)}")

@cache
def generate_line_chart(df, x_axis, y_axis, dark_mode, selected_category):
    try:
        if not x_axis or not y_axis or x_axis not in df.columns or y_axis not in df.columns:
            return px.scatter(title="Please select X and Y axes for the Line Chart")
        if len(df) > 500:
            df = df.sample(500, random_state=42)
        fig = px.line(df, x=x_axis, y=y_axis, template='plotly_dark' if dark_mode else 'plotly', title=f"Line: {x_axis} vs {y_axis}")
        if selected_category:
            fig.add_scatter(x=[selected_category], y=[df[df[x_axis] == selected_category][y_axis].iloc[0]], mode='markers', marker=dict(size=15, color='#00d4ff'), showlegend=False)
        return fig
    except Exception as e:
        logger.error(f"Line chart error: {e}")
        return px.scatter(title=f"Error generating line chart: {str(e)}")

@cache
def generate_correlation_heatmap(df, dark_mode):
    try:
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numerical_cols) < 2:
            return px.scatter(title="Not enough numerical columns for correlation")
        if len(df) > 500:
            df = df.sample(500, random_state=42)
        corr_matrix = df[numerical_cols].corr()
        fig = ff.create_annotated_heatmap(z=corr_matrix.values, x=numerical_cols, y=numerical_cols, colorscale='Plasma' if dark_mode else 'Viridis', showscale=True)
        fig.update_layout(title="Correlation Heatmap", template='plotly_dark' if dark_mode else 'plotly')
        return fig
    except Exception as e:
        logger.error(f"Correlation heatmap error: {e}")
        return px.scatter(title=f"Error generating correlation heatmap: {str(e)}")

@cache
def generate_trend_chart(df, x_axis, y_axis, dark_mode):
    try:
        if not x_axis or not y_axis or x_axis not in df.columns or y_axis not in df.columns:
            return px.scatter(title="Select a date column (X-axis) and Y-axis for trend analysis")
        df[x_axis] = pd.to_datetime(df[x_axis], errors='coerce')
        if df[x_axis].isna().all():
            return px.scatter(title="Invalid date column format")
        if len(df) > 500:
            df = df.sample(500, random_state=42)
        trend_data = df.groupby(x_axis)[y_axis].sum().reset_index()
        fig = px.line(trend_data, x=x_axis, y=y_axis, template='plotly_dark' if dark_mode else 'plotly')
        fig.update_layout(title=f"Trend of {y_axis} Over Time")
        return fig
    except Exception as e:
        logger.error(f"Trend chart error: {e}")
        return px.scatter(title=f"Error generating trend chart: {str(e)}")

@cache
def generate_forecast(df, x_axis, y_axis, periods=30, dark_mode=True):
    try:
        if not x_axis or x_axis not in df.columns:
            return px.scatter(title="Please select a valid date column for X-axis")
        if not y_axis or y_axis not in df.columns:
            return px.scatter(title="Please select a valid numerical column for Y-axis")
        if not pd.api.types.is_numeric_dtype(df[y_axis]):
            return px.scatter(title=f"Y-axis '{y_axis}' is not numerical")
        df[x_axis] = pd.to_datetime(df[x_axis], errors='coerce')
        if df[x_axis].isna().all():
            return px.scatter(title="Invalid date column format")
        forecast_data = df[[x_axis, y_axis]].rename(columns={x_axis: 'ds', y_axis: 'y'}).dropna()
        if len(forecast_data) < 2:
            return px.scatter(title="Not enough data for forecasting")
        forecast_data = forecast_data.groupby('ds')['y'].sum().reset_index()
        train_size = int(len(forecast_data) * 0.8)
        train_data = forecast_data[:train_size]
        test_data = forecast_data[train_size:]
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, changepoint_prior_scale=0.05, seasonality_prior_scale=10.0)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        potential_regressors = [col for col in numerical_cols if col != y_axis and col in df.columns]
        regressors = [col for col in potential_regressors if df[col].notna().sum() > 0]
        for col in regressors:
            model.add_regressor(col)
        if regressors:
            train_data_with_regressors = df[[x_axis, y_axis] + regressors].rename(columns={x_axis: 'ds', y_axis: 'y'})
            train_data_with_regressors = train_data_with_regressors[train_data_with_regressors['ds'].isin(train_data['ds'])].groupby('ds').agg({'y': 'sum', **{regressor: 'mean' for regressor in regressors}}).reset_index()
        else:
            train_data_with_regressors = train_data
        model.fit(train_data_with_regressors)
        future = model.make_future_dataframe(periods=periods + len(test_data), freq='D')
        if regressors:
            for regressor in regressors:
                future[regressor] = df[regressor].mean()
        forecast = model.predict(future)
        test_forecast = forecast[forecast['ds'].isin(test_data['ds'])]
        test_actual = test_data['y'].values
        test_predicted = test_forecast['yhat'].values
        mae = mean_absolute_error(test_actual, test_predicted)
        rmse = np.sqrt(mean_squared_error(test_actual, test_predicted))
        mape = np.mean(np.abs((test_actual - test_predicted) / test_actual)) * 100
        accuracy_message = f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['y'], mode='lines', name='Actual', line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='#ff7f0e')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line=dict(color='rgba(0,0,0,0)'), name='Confidence Interval'))
        fig.update_layout(title=f"Forecast of {y_axis} for Next {periods} Days ({accuracy_message})", template='plotly_dark' if dark_mode else 'plotly', xaxis_title="Date", yaxis_title=y_axis, hovermode='x unified')
        return fig
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return px.scatter(title=f"Error in forecasting: {str(e)}")

@cache
def generate_feature_importance(df, x_axis, y_axis, dark_mode):
    try:
        if not y_axis or y_axis not in df.columns or df[y_axis].dtype not in ['int64', 'float64']:
            return px.scatter(title="Select a numerical Y-axis for feature importance")
        features = df.select_dtypes(include=['number']).columns.tolist()
        features = [col for col in features if col != y_axis]
        if not features:
            return px.scatter(title="Not enough numerical features for analysis")
        if len(df) > 500:
            df = df.sample(500, random_state=42)
        X = df[features].fillna(0)
        y = df[y_axis].fillna(0)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        importances = model.feature_importances_
        fig = px.bar(x=importances, y=features, orientation='h', template='plotly_dark' if dark_mode else 'plotly')
        fig.update_layout(title=f"Feature Importance for {y_axis}", xaxis_title="Importance", yaxis_title="Feature")
        return fig
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        return px.scatter(title=f"Error generating feature importance: {str(e)}")

@cache
def generate_smart_insights(df):
    try:
        insights = []
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        for cat_col in categorical_cols:
            for num_col in numerical_cols:
                if any(k in num_col.lower() for k in ['revenue', 'sales', 'amount']):
                    top_performers = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(3)
                    insights.append(html.P(f"🔹 Top 3 {cat_col} by {num_col}:", className="text-light"))
                    insights.append(html.Ul([html.Li(f"{idx}: {val:.2f}") for idx, val in top_performers.items()], className="text-light"))
        if datetime_cols:
            date_col = datetime_cols[0]
            for num_col in numerical_cols:
                if any(k in num_col.lower() for k in ['revenue', 'sales', 'amount']):
                    df['month'] = df[date_col].dt.month
                    monthly_trend = df.groupby('month')[num_col].sum()
                    max_month = monthly_trend.idxmax()
                    min_month = monthly_trend.idxmin()
                    insights.append(html.P(f"🔹 {num_col} Trend:", className="text-light"))
                    insights.append(html.P(f"Highest in month {max_month}: {monthly_trend[max_month]:.2f}", className="text-light"))
                    insights.append(html.P(f"Lowest in month {min_month}: {monthly_trend[min_month]:.2f}", className="text-light"))
        return insights if insights else ["No smart insights available"]
    except Exception as e:
        logger.error(f"Smart insights error: {e}")
        return [f"Error generating smart insights: {str(e)}"]

def generate_kpi_cards(df):
    try:
        kpi_cards = []
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        revenue_col = next((col for col in numerical_cols if any(k in col.lower() for k in ['revenue', 'amount'])), None)
        if revenue_col:
            total_revenue = df[revenue_col].sum()
            kpi_cards.append(dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H6(f"Total {revenue_col.replace('_', ' ').title()}", className="card-title"),
                        html.H4(f"${total_revenue:,.2f}", className="card-text")
                    ])
                ], className="kpi-card"),
                width=3
            ))
        if revenue_col and 'order_id' in df.columns:
            avg_order_value = df.groupby('order_id')[revenue_col].sum().mean()
            kpi_cards.append(dbc.Col(dbc.Card([dbc.CardBody([html.H6("Avg Order Value", className="card-title"), html.H4(f"${avg_order_value:,.2f}", className="card-text")])], className="kpi-card"), width=3))
        product_col = next((col for col in df.columns if 'product' in col.lower()), None)
        if product_col and revenue_col:
            top_product = df.groupby(product_col)[revenue_col].sum().idxmax()
            top_product_revenue = df.groupby(product_col)[revenue_col].sum().max()
            kpi_cards.append(dbc.Col(dbc.Card([dbc.CardBody([html.H6("Top Product", className="card-title"), html.H4(f"{top_product}", className="card-text"), html.P(f"{revenue_col.title()}: ${top_product_revenue:,.2f}", className="")])], className="kpi-card"), width=3))
        store_col = next((col for col in df.columns if 'store' in col.lower()), None)
        if store_col and revenue_col:
            top_store = df.groupby(store_col)[revenue_col].sum().idxmax()
            top_store_revenue = df.groupby(store_col)[revenue_col].sum().max()
            kpi_cards.append(dbc.Col(dbc.Card([dbc.CardBody([html.H6("Top Store", className="card-title"), html.H4(f"{top_store}", className="card-text"), html.P(f"{revenue_col.title()}: ${top_store_revenue:,.2f}", className="")])], className="kpi-card"), width=3))
        return kpi_cards
    except Exception as e:
        logger.error(f"KPI cards error: {e}")
        return [dbc.Col(html.P(f"Error generating KPI cards: {str(e)}", className="text-danger"), width=12)]

def perform_scenario_analysis(df, scenario_column, scenario_adjustment):
    try:
        if not scenario_column or scenario_column not in df.columns or not pd.api.types.is_numeric_dtype(df[scenario_column]):
            return "Select a numerical column for scenario analysis", []
        df_scenario = df.copy()
        df_scenario[scenario_column] = df_scenario[scenario_column] * (1 + scenario_adjustment / 100)
        revenue_col = next((col for col in df.columns if any(k in col.lower() for k in ['revenue', 'amount'])), None)
        price_col = next((col for col in df.columns if 'price' in col.lower()), None)
        sales_col = next((col for col in df.columns if 'sales' in col.lower()), None)
        if revenue_col and price_col and sales_col:
            df_scenario[revenue_col] = df_scenario[price_col] * df_scenario[sales_col]
        original_metrics = {}
        scenario_metrics = {}
        if revenue_col:
            original_metrics[f'Total {revenue_col.title()}'] = df[revenue_col].sum()
            scenario_metrics[f'Total {revenue_col.title()}'] = df_scenario[revenue_col].sum()
        if revenue_col and 'order_id' in df.columns:
            original_metrics['Avg Order Value'] = df.groupby('order_id')[revenue_col].sum().mean()
            scenario_metrics['Avg Order Value'] = df_scenario.groupby('order_id')[revenue_col].sum().mean()
        comparison_table = dash_table.DataTable(
            columns=[{"name": "Metric", "id": "metric"}, {"name": "Original", "id": "original"}, {"name": "Scenario", "id": "scenario"}, {"name": "Change (%)", "id": "change"}],
            data=[{"metric": metric, "original": f"${original_metrics[metric]:,.2f}", "scenario": f"${scenario_metrics[metric]:,.2f}", "change": f"{((scenario_metrics[metric] - original_metrics[metric]) / original_metrics[metric] * 100):.2f}%"} for metric in original_metrics.keys()],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'backgroundColor': '#1f2a44', 'color': 'white'},
            style_header={'backgroundColor': '#1f2a44', 'color': 'white', 'fontWeight': 'bold'},
        )
        return None, comparison_table
    except Exception as e:
        logger.error(f"Scenario analysis error: {e}")
        return f"Error performing scenario analysis: {str(e)}", []

@cache
def generate_ai_recommendations(df):
    try:
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numerical_cols) < 2:
            return ["Not enough numerical columns for AI recommendations"]
        features = df[numerical_cols].dropna()
        if len(features) < 3:
            return [f"Not enough samples ({len(features)}) for recommendations (need at least 3)"]
        recommendations = []
        revenue_col = next((col for col in df.columns if any(k in col.lower() for k in ['revenue', 'amount'])), None)
        stock_col = next((col for col in df.columns if 'stock' in col.lower()), None)
        product_col = next((col for col in df.columns if 'product' in col.lower()), None)
        store_col = next((col for col in df.columns if 'store' in col.lower()), None)
        if revenue_col and stock_col and product_col and store_col:
            top_products = df.groupby([product_col, store_col])[revenue_col].sum().reset_index()
            top_products = top_products.sort_values(by=revenue_col, ascending=False).head(3)
            for _, row in top_products.iterrows():
                recommendation = f"Increase stock for {row[product_col]} in {row[store_col]} to potentially boost {revenue_col.lower()} by 10%"
                recommendations.append(recommendation)
        return recommendations if recommendations else ["No AI recommendations available"]
    except Exception as e:
        logger.error(f"AI recommendations error: {e}")
        return [f"Error generating AI recommendations: {str(e)}"]

def generate_executive_summary(df, kpi_cards, smart_insights, ai_recommendations):
    try:
        if pdfkit is None:
            return None
        html_content = """
        <html>
        <head>
            <title>Executive Summary Report</title>
            <style>
                body { font-family: Arial, sans-serif; background-color: #f4f4f4; color: #333; }
                h1, h2 { color: #1f2a44; }
                .section { margin: 20px 0; padding: 10px; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .kpi { display: flex; justify-content: space-around; }
                .kpi-card { background-color: #2e3b55; color: #fff; padding: 10px; border-radius: 5px; width: 20%; text-align: center; }
                ul { list-style-type: disc; padding-left: 20px; }
            </style>
        </head>
        <body>
            <h1>Executive Summary Report</h1>
            <div class="section">
                <h2>Key Performance Indicators</h2>
                <div class="kpi">
        """
        for card in kpi_cards:
            card_html = card.children[0].children[0]
            title = card_html.children[0].children
            value = card_html.children[1].children
            html_content += f'<div class="kpi-card"><h3>{title}</h3><p>{value}</p></div>'
        html_content += """
                </div>
            </div>
            <div class="section">
                <h2>Smart Insights</h2>
                <ul>
        """
        for insight in smart_insights:
            if isinstance(insight, html.P):
                html_content += f"<li>{insight.children}</li>"
            elif isinstance(insight, html.Ul):
                for li in insight.children:
                    html_content += f"<li>{li.children}</li>"
        html_content += """
                </ul>
            </div>
            <div class="section">
                <h2>AI Recommendations</h2>
                <ul>
        """
        for rec in ai_recommendations:
            html_content += f"<li>{rec}</li>"
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        pdf_buffer = io.BytesIO()
        pdfkit.from_string(html_content, pdf_buffer)
        pdf_buffer.seek(0)
        return dict(content=base64.b64encode(pdf_buffer.getvalue()).decode(), filename="executive_summary.pdf", type="application/pdf", base64=True)
    except Exception as e:
        logger.error(f"Executive summary error: {e}")
        return None

sidebar = dbc.Col([
    dbc.Card([
        dbc.CardBody([
            html.H4("Controls", className="card-title text-light"),
            dcc.Upload(id='upload-data', children=html.Button('Upload File 📁', className="btn btn-outline-cyan btn-block mb-3"), multiple=False, accept=".csv,.xlsx"),
            html.Div(id='cleaning-message', className="mb-3 text-center text-light"),
            html.Div(id='upload-message', className="mb-3 text-center text-light"),
            html.H5("Data Analysis", className="card-title text-light mt-3"),
            dcc.Dropdown(id='analysis-x-axis', placeholder="Select X-axis", className="mb-3 dropdown-purple"),
            dcc.Dropdown(id='x-axis-filter-values', placeholder="Select X-axis filter values", value=[], multi=True, className="mb-3 dropdown-purple"),
            dcc.Dropdown(id='analysis-y-axis', placeholder="Select Y-axis (Multiple)", multi=True, className="mb-3 dropdown-purple"),
            dcc.Dropdown(id='chart-type', options=[
                {'label': 'Bar Chart', 'value': 'bar'},
                {'label': 'Pie Chart', 'value': 'pie'},
                {'label': 'Scatter Plot', 'value': 'scatter'},
                {'label': 'Line Chart', 'value': 'line'},
                {'label': 'Heatmap', 'value': 'heatmap'},
                {'label': 'Box Plot', 'value': 'box'},
                {'label': 'Regression', 'value': 'regression'}
            ], value='bar', className="mb-3 dropdown-purple"),
            html.H5("Data Prediction", className="card-title text-light mt-3"),
            dcc.Dropdown(id='prediction-x-axis', placeholder="Select Date Column (X-axis)", className="mb-3 dropdown-purple"),
            dcc.Dropdown(id='prediction-y-axis', placeholder="Select Numerical Column (Y-axis)", className="mb-3 dropdown-purple"),
            dcc.Dropdown(id='column-filter', placeholder="Select a column to filter", className="mb-3 dropdown-purple"),
            dcc.Dropdown(id='value-filter', placeholder="Select values to filter", value=[], multi=True, className="mb-3 dropdown-purple"),
            dbc.Switch(id='dark-mode-toggle', label='Dark Mode', value=True, className="mb-3 text-light"),
            html.Button("Download Data 📥", id="download-button", className="btn btn-outline-cyan btn-block mb-3"),
            html.Button("Clear Selection", id="clear-selection", className="btn btn-outline-red btn-block mb-3"),
            html.H5("What-If Analysis", className="card-title text-light mt-3"),
            html.Label("Adjust Numerical Values (% Change)", className="text-light"),
            dcc.Dropdown(id='what-if-column', placeholder="Select a numerical column", className="mb-3 dropdown-purple"),
            dcc.Slider(id='what-if-adjust', min=-50, max=50, step=5, value=0, marks={i: f"{i}%" for i in range(-50, 51, 25)}),
            html.H5("Scenario Analysis", className="card-title text-light mt-3"),
            dcc.Dropdown(id='scenario-column', placeholder="Select a column for scenario", className="mb-3 dropdown-purple"),
            dcc.Slider(id='scenario-adjust', min=-50, max=50, step=5, value=0, marks={i: f"{i}%" for i in range(-50, 51, 25)}),
            html.Div(id='scenario-results'),
            html.H5("Smart Insights", className="card-title text-light mt-3"),
            html.Div(id='smart-insights', style={'maxHeight': '200px', 'overflowY': 'auto'}),
            html.H5("AI Recommendations", className="card-title text-light mt-3"),
            html.Div(id='ai-recommendations', style={'maxHeight': '200px', 'overflowY': 'auto'}),
            html.Button("Export Recommendations 📜", id="export-recommendations", className="btn btn-outline-cyan btn-block mb-3"),
            dcc.Download(id="download-recommendations"),
            html.H5("Collaboration", className="card-title text-light mt-3"),
            dcc.Textarea(id='comment-input', placeholder="Add a comment...", style={'width': '100%', 'height': 50}, className="mb-2"),
            html.Button("Submit Comment", id="submit-comment", className="btn btn-outline-cyan btn-block mb-3"),
            html.Div(id='comments-section', style={'maxHeight': '200px', 'overflowY': 'auto'}),
            html.H5("Summary Statistics", className="card-title text-light mt-3"),
            html.Div(id='summary-stats'),
            html.H5("Export Report", className="card-title text-light mt-3"),
            html.Button("Generate Executive Summary 📄", id="export-report", className="btn btn-outline-cyan btn-block mb-3"),
            dcc.Download(id="download-report"),
        ])
    ], className="bg-card-dark border-0 shadow-sm")
], width=3, className="sidebar collapse show", id="sidebar")

main_content = dbc.Col([
    dbc.NavbarSimple(
        brand="📊 Extraordinary Data Analysis Dashboard",
        color="dark",
        dark=True,
        className="mb-4 navbar-dark",
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="#", className="text-light")),
            dbc.NavItem(dbc.NavLink("About", href="#", className="text-light")),
        ]
    ),
    dcc.Store(id='selected-category'),
    dcc.Store(id='filtered-data', data=None),
    dbc.Row(id='kpi-cards', className="mb-4"),
    dbc.Card([
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab(label="Main Chart", tab_id="main-tab", children=[
                    dcc.Loading(id="loading-main-chart", type="default", children=dcc.Graph(id='main-chart', style={'height': '400px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']}))
                ]),
                dbc.Tab(label="Pie Chart", tab_id="pie-tab", children=[
                    dcc.Loading(id="loading-pie-chart", type="default", children=dcc.Graph(id='pie-chart', style={'height': '400px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']})),
                    html.Div([
                        html.Label("Select Column for Pie Chart:", className="text-light mr-2"),
                        dcc.Dropdown(id='pie-names-column', placeholder="Select a column", className="dropdown-purple", style={'width': '200px', 'display': 'inline-block'})
                    ], style={'textAlign': 'center', 'marginTop': '10px'})
                ]),
                dbc.Tab(label="Line Chart", tab_id="line-tab", children=[
                    dcc.Loading(id="loading-line-chart", type="default", children=dcc.Graph(id='line-chart', style={'height': '400px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']})),
                    html.Div([
                        html.Div([
                            html.Label("Y-Axis:", className="text-light mr-2"),
                            dcc.Dropdown(id='line-y-axis', placeholder="Select Y-axis", className="dropdown-purple", style={'width': '200px', 'display': 'inline-block'})
                        ], style={'position': 'absolute', 'bottom': '10px', 'left': '10px'}),
                        html.Div([
                            html.Label("X-Axis:", className="text-light mr-2"),
                            dcc.Dropdown(id='line-x-axis', placeholder="Select X-axis", className="dropdown-purple", style={'width': '200px', 'display': 'inline-block'})
                        ], style={'position': 'absolute', 'bottom': '10px', 'right': '10px'})
                    ], style={'position': 'relative', 'height': '50px'})
                ]),
                dbc.Tab(label="Correlation", tab_id="correlation-tab", children=[
                    dcc.Loading(id="loading-correlation-heatmap", type="default", children=dcc.Graph(id='correlation-heatmap', style={'height': '400px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']}))
                ]),
                dbc.Tab(label="Trends", tab_id="trends-tab", children=[
                    dcc.Loading(id="loading-trend-chart", type="default", children=dcc.Graph(id='trend-chart', style={'height': '400px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']}))
                ]),
                dbc.Tab(label="Forecast", tab_id="forecast-tab", children=[
                    dcc.Loading(id="loading-forecast-chart", type="default", children=dcc.Graph(id='forecast-chart', style={'height': '400px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']}))
                ]),
                dbc.Tab(label="Feature Importance", tab_id="feature-importance-tab", children=[
                    dcc.Loading(id="loading-feature-importance", type="default", children=dcc.Graph(id='feature-importance', style={'height': '400px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']}))
                ]),
            ], id="tabs", active_tab="main-tab")
        ])
    ], className="bg-card-dark border-0 shadow-sm mb-4"),
    dbc.Card([
        dbc.CardBody([
            html.H5("Data Table", className="card-title text-light mb-3"),
            dash_table.DataTable(
                id='data-table',
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'backgroundColor': '#1f2a44', 'color': 'white'},
                style_header={'backgroundColor': '#1f2a44', 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#2e3b55'},
                    {'if': {'state': 'selected'}, 'backgroundColor': '#00d4ff', 'color': 'black'}
                ],
                sort_action='native',
                filter_action='native',
                export_format='csv',
                export_headers='display'
            )
        ])
    ], className="bg-card-dark border-0 shadow-sm mb-4"),
    dcc.Download(id="download-data"),
    html.Button("Download Analysis Charts 📊", id="download-analysis-charts", className="btn btn-outline-cyan btn-block mb-3"),
    dcc.Download(id="download-analysis-charts-zip"),
    html.Button("Download Prediction Charts 📈", id="download-prediction-charts", className="btn btn-outline-cyan btn-block mb-3"),
    dcc.Download(id="download-prediction-charts-zip"),
], width=9)

app.layout = dbc.Container([
    dbc.Row([
        sidebar,
        main_content
    ])
], fluid=True)

@app.callback(
    [Output('analysis-x-axis', 'options'),
     Output('analysis-y-axis', 'options'),
     Output('prediction-x-axis', 'options'),
     Output('prediction-y-axis', 'options'),
     Output('column-filter', 'options'),
     Output('what-if-column', 'options'),
     Output('scenario-column', 'options'),
     Output('pie-names-column', 'options'),
     Output('line-x-axis', 'options'),
     Output('line-y-axis', 'options'),
     Output('upload-message', 'children'),
     Output('cleaning-message', 'children'),
     Output('filtered-data', 'data'),
     Output('data-table', 'data'),
     Output('data-table', 'columns'),
     Output('analysis-x-axis', 'value', allow_duplicate=True),
     Output('analysis-y-axis', 'value', allow_duplicate=True)],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')],
    prevent_initial_call=True
)
def update_dropdowns(contents, filename):
    if contents is None:
        return [], [], [], [], [], [], [], [], [], [], "Please upload a file", "", None, [], [], None, []

    df, messages = parse_contents(contents, filename)
    if df is None:
        return [], [], [], [], [], [], [], [], [], [], messages, "", None, [], [], None, []

    data_store.df = df
    data_store.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    columns = [{'label': col, 'value': col} for col in df.columns]
    numerical_cols = [{'label': col, 'value': col} for col in df.select_dtypes(include=['number']).columns]
    datetime_cols = [{'label': col, 'value': col} for col in df.select_dtypes(include=['datetime']).columns]

    table_columns = [{"name": col, "id": col} for col in df.columns]
    table_data = df.to_dict('records')

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        default_x_axis = categorical_cols[0]
    elif datetime_cols:
        default_x_axis = datetime_cols[0]['value']
    else:
        default_x_axis = df.columns[0]

    default_y_axis = numerical_cols[0]['value'] if numerical_cols else None

    return (columns, columns, datetime_cols, numerical_cols, columns, numerical_cols, numerical_cols,
            columns, columns, numerical_cols, f"File {filename} uploaded successfully", messages,
            df.to_dict('records'), table_data, table_columns, default_x_axis, [default_y_axis] if default_y_axis else [])

@app.callback(
    Output('value-filter', 'options'),
    [Input('column-filter', 'value')],
    [State('filtered-data', 'data')]
)
def update_value_filter(column, data):
    if not column or not data:
        return []
    df = pd.DataFrame(data)
    if column not in df.columns:
        return []
    unique_values = df[column].dropna().unique()
    return [{'label': str(val), 'value': str(val)} for val in unique_values]

@app.callback(
    Output('x-axis-filter-values', 'options'),
    [Input('analysis-x-axis', 'value')],
    [State('filtered-data', 'data')]
)
def update_x_axis_filter_values(x_axis, data):
    if not x_axis or not data:
        return []
    df = pd.DataFrame(data)
    if x_axis not in df.columns:
        return []
    unique_values = df[x_axis].dropna().unique()
    return [{'label': str(val), 'value': str(val)} for val in unique_values]

@app.callback(
    [Output('kpi-cards', 'children'),
     Output('smart-insights', 'children'),
     Output('filtered-data', 'data', allow_duplicate=True),
     Output('data-table', 'data', allow_duplicate=True)],
    [Input('filtered-data', 'data'),
     Input('value-filter', 'value'),
     Input('column-filter', 'value'),
     Input('analysis-x-axis', 'value'),
     Input('what-if-column', 'value'),
     Input('what-if-adjust', 'value'),
     Input('x-axis-filter-values', 'value')],
    prevent_initial_call=True
)
def update_kpi_and_insights(data, filter_values, filter_column, analysis_x_axis, what_if_column, what_if_adjust, x_axis_filter_values):
    if not data:
        return [], ["Please upload data to see insights"], None, []
    
    try:
        df = pd.DataFrame(data)
        logger.info(f"Initial DataFrame shape: {df.shape}")
        logger.info(f"Initial unique values in {analysis_x_axis}: {df[analysis_x_axis].unique() if analysis_x_axis in df.columns else 'N/A'}")
        
        # Apply column filter (if any)
        if filter_column and filter_values:
            df = df[df[filter_column].isin(filter_values)]
            logger.info(f"After column filter ({filter_column} in {filter_values}): DataFrame shape: {df.shape}")
            if df.empty:
                return [], ["No data available after filtering"], None, []
        
        # Apply X-axis filter (if any)
        if analysis_x_axis and x_axis_filter_values and analysis_x_axis in df.columns:
            df = df[df[analysis_x_axis].isin(x_axis_filter_values)]
            logger.info(f"After X-axis filter ({analysis_x_axis} in {x_axis_filter_values}): DataFrame shape: {df.shape}")
            if df.empty:
                return [], ["No data available after X-axis filtering"], None, []
        
        # Apply what-if analysis (if any)
        if what_if_column and what_if_adjust and what_if_column in df.columns:
            if pd.api.types.is_numeric_dtype(df[what_if_column]):
                df[what_if_column] = df[what_if_column] * (1 + what_if_adjust / 100)
        
        data_store.filtered_df = df
        kpi_cards = generate_kpi_cards(df)
        smart_insights = generate_smart_insights(df)
        table_data = df.to_dict('records')
        logger.info(f"Final filtered DataFrame shape: {df.shape}")
        return kpi_cards, smart_insights, df.to_dict('records'), table_data
    except Exception as e:
        logger.error(f"Error in update_kpi_and_insights: {str(e)}")
        return [], [f"Error processing data: {str(e)}"], None, []

@app.callback(
    [Output('main-chart', 'figure'),
     Output('pie-chart', 'figure'),
     Output('line-chart', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('trend-chart', 'figure'),
     Output('forecast-chart', 'figure'),
     Output('feature-importance', 'figure')],
    [Input('tabs', 'active_tab'),
     Input('filtered-data', 'data'),
     Input('analysis-x-axis', 'value'),
     Input('analysis-y-axis', 'value'),
     Input('chart-type', 'value'),
     Input('dark-mode-toggle', 'value'),
     Input('selected-category', 'data'),
     Input('prediction-x-axis', 'value'),
     Input('prediction-y-axis', 'value'),
     Input('pie-names-column', 'value'),
     Input('line-x-axis', 'value'),
     Input('line-y-axis', 'value')]
)
def update_charts(active_tab, data, x_axis, y_axes, chart_type, dark_mode, selected_category, pred_x_axis, pred_y_axis, pie_names_col, line_x_axis, line_y_axis):
    empty_fig = px.scatter(title="Please upload data")
    if not data:
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
    
    try:
        df = pd.DataFrame(data)
        if df.empty or df is None:
            empty_fig = px.scatter(title="No data available")
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
        
        main_fig = px.scatter(title="Select X and Y axes")
        pie_fig = px.scatter(title="Select a column for the Pie Chart")
        line_fig = px.scatter(title="Select X and Y axes for the Line Chart")
        heatmap_fig = px.scatter(title="Select numerical columns")
        trend_fig = px.scatter(title="Select a date column for trends")
        forecast_fig = px.scatter(title="Select a date and numerical column for forecasting")
        feature_fig = px.scatter(title="Select a numerical Y-axis for feature importance")
        
        if active_tab == "main-tab" and x_axis and y_axes:
            main_fig = generate_main_chart(df, x_axis, y_axes if y_axes else [], chart_type, dark_mode, selected_category)
        elif active_tab == "pie-tab" and pie_names_col:
            pie_fig = generate_pie_chart(df, pie_names_col, dark_mode, selected_category)
        elif active_tab == "line-tab" and line_x_axis and line_y_axis:
            line_fig = generate_line_chart(df, line_x_axis, line_y_axis, dark_mode, selected_category)
        elif active_tab == "correlation-tab":
            heatmap_fig = generate_correlation_heatmap(df, dark_mode)
        elif active_tab == "trends-tab" and x_axis and y_axes:
            y_axis = y_axes[0] if y_axes else None
            trend_fig = generate_trend_chart(df, x_axis, y_axis, dark_mode)
        elif active_tab == "forecast-tab" and pred_x_axis and pred_y_axis:
            forecast_fig = generate_forecast(df, pred_x_axis, pred_y_axis, dark_mode=dark_mode)
        elif active_tab == "feature-importance-tab" and y_axes:
            y_axis = y_axes[0] if y_axes else None
            feature_fig = generate_feature_importance(df, x_axis, y_axis, dark_mode)
        
        return main_fig, pie_fig, line_fig, heatmap_fig, trend_fig, forecast_fig, feature_fig
    except Exception as e:
        logger.error(f"Error in update_charts: {str(e)}")
        error_fig = px.scatter(title=f"Chart update error: {str(e)}")
        return error_fig, error_fig, error_fig, error_fig, error_fig, error_fig, error_fig

@app.callback(
    Output('selected-category', 'data'),
    [Input('main-chart', 'clickData')],
    [State('analysis-x-axis', 'value')]
)
def update_selected_category(click_data, x_axis):
    if not click_data or not x_axis:
        return None
    return click_data['points'][0]['x']

@app.callback(
    Output('download-data', 'data'),
    [Input('download-button', 'n_clicks')],
    [State('filtered-data', 'data')],
    prevent_initial_call=True
)
def download_data(n_clicks, data):
    if not data:
        return None
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return dict(content=csv_buffer.getvalue(), filename="filtered_data.csv")

@app.callback(
    Output('download-analysis-charts-zip', 'data'),
    [Input('download-analysis-charts', 'n_clicks')],
    [State('main-chart', 'figure'),
     State('pie-chart', 'figure'),
     State('line-chart', 'figure'),
     State('correlation-heatmap', 'figure')],
    prevent_initial_call=True
)
def download_analysis_charts(n_clicks, main_fig, pie_fig, line_fig, heatmap_fig):
    if not n_clicks:
        return None

    charts = [
        ('main_chart.png', main_fig),
        ('pie_chart.png', pie_fig),
        ('line_chart.png', line_fig),
        ('correlation_heatmap.png', heatmap_fig)
    ]

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, fig in charts:
            if fig and 'data' in fig and fig['data']:
                try:
                    img_bytes = pio.to_image(fig, format='png')
                    zip_file.writestr(filename, img_bytes)
                except Exception as e:
                    logger.error(f"Error saving {filename}: {str(e)}")
                    continue

    zip_buffer.seek(0)
    return dict(
        content=base64.b64encode(zip_buffer.getvalue()).decode(),
        filename="analysis_charts.zip",
        type="application/zip",
        base64=True
    )

@app.callback(
    Output('download-prediction-charts-zip', 'data'),
    [Input('download-prediction-charts', 'n_clicks')],
    [State('trend-chart', 'figure'),
     State('forecast-chart', 'figure'),
     State('feature-importance', 'figure')],
    prevent_initial_call=True
)
def download_prediction_charts(n_clicks, trend_fig, forecast_fig, feature_fig):
    if not n_clicks:
        return None

    charts = [
        ('trend_chart.png', trend_fig),
        ('forecast_chart.png', forecast_fig),
        ('feature_importance.png', feature_fig)
    ]

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, fig in charts:
            if fig and 'data' in fig and fig['data']:
                try:
                    img_bytes = pio.to_image(fig, format='png')
                    zip_file.writestr(filename, img_bytes)
                except Exception as e:
                    logger.error(f"Error saving {filename}: {str(e)}")
                    continue

    zip_buffer.seek(0)
    return dict(
        content=base64.b64encode(zip_buffer.getvalue()).decode(),
        filename="prediction_charts.zip",
        type="application/zip",
        base64=True
    )

@app.callback(
    [Output('analysis-x-axis', 'value'),
     Output('analysis-y-axis', 'value'),
     Output('chart-type', 'value'),
     Output('column-filter', 'value'),
     Output('value-filter', 'value'),
     Output('what-if-column', 'value'),
     Output('what-if-adjust', 'value'),
     Output('scenario-column', 'value'),
     Output('scenario-adjust', 'value'),
     Output('x-axis-filter-values', 'value')],
    [Input('clear-selection', 'n_clicks')]
)
def clear_selection(n_clicks):
    if n_clicks:
        return None, [], 'bar', None, [], None, 0, None, 0, []
    return dash.no_update

@app.callback(
    [Output('scenario-results', 'children')],
    [Input('scenario-column', 'value'),
     Input('scenario-adjust', 'value')],
    [State('filtered-data', "data")]
)
def update_scenario_analysis(scenario_column, scenario_adjust, data):
    if not data or not scenario_column:
        return ["Select a column for scenario analysis"]
    df = pd.DataFrame(data)
    error, table = perform_scenario_analysis(df, scenario_column, scenario_adjust)
    if error:
        return [error]
    return [table]

@app.callback(
    Output('ai-recommendations', 'children'),
    [Input('filtered-data', 'data')]
)
def update_recommendations(data):
    if not data:
        return ["Please upload data"]
    
    df = pd.DataFrame(data)
    ai_recommendations = generate_ai_recommendations(df)
    return ai_recommendations

@app.callback(
    Output('download-recommendations', 'data'),
    [Input('export-recommendations', "n_clicks")],
    [State('ai-recommendations', 'children')],
    prevent_initial_call=True
)
def export_recommendations(n_clicks, recommendations):
    if not recommendations:
        return None
    content = "\n".join([rec if isinstance(rec, str) else rec.children for rec in recommendations])
    return dict(content=content, filename="recommendations.txt")

@app.callback(
    [Output('comments-section', 'children'),
     Output('comment-input', 'value')],
    [Input('submit-comment', 'n_clicks')],
    [State('comment-input', 'value')]
)
def update_comments(n_clicks, comment):
    if not n_clicks or not comment:
        return data_store.comments, ""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_store.comments.append(f"{timestamp}: {comment}")
    return data_store.comments, ""

@app.callback(
    Output('summary-stats', 'children'),
    [Input('filtered-data', 'data')]
)
def update_summary_stats(data):
    if not data:
        return "Please upload data"
    df = pd.DataFrame(data)
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numerical_cols:
        return "No numerical columns available for summary statistics"
    stats = df[numerical_cols].describe().reset_index()
    return dash_table.DataTable(
        data=stats.to_dict('records'),
        columns=[{"name": i, "id": i} for i in stats.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'backgroundColor': '#1f2a44', 'color': 'white'},
        style_header={'backgroundColor': '#1f2a44', 'color': 'white', 'fontWeight': 'bold'}
    )

@app.callback(
    Output('download-report', 'data'),
    [Input('export-report', 'n_clicks')],
    [State('filtered-data', 'data'),
     State('kpi-cards', 'children'),
     State('smart-insights', 'children'),
     State('ai-recommendations', 'children')],
    prevent_initial_call=True
)
def export_executive_summary(n_clicks, data, kpi_cards, smart_insights, ai_recommendations):
    if not data:
        return None
    df = pd.DataFrame(data)
    return generate_executive_summary(df, kpi_cards, smart_insights, ai_recommendations)

if __name__ == '__main__':
    app.run_server(debug=True)