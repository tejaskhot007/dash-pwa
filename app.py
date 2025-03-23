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
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash App with a dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="📊 Advanced Data Analysis Dashboard",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "theme-color", "content": "#343a40"},
        {"name": "mobile-web-app-capable", "content": "yes"},
        {"name": "apple-mobile-web-app-status-bar-style", "content": "black-translucent"},
    ],
)
server = app.server

# Global data store
class DataStore:
    def __init__(self):
        self.df = None
        self.last_updated = None
        self.filtered_df = None
        self.raw_df = None  # Store the raw DataFrame before cleaning
        self.original_row_count = None  # Store the original row count for comparison

data_store = DataStore()

# Data cleaning function
def clean_data(df):
    try:
        # Store the raw DataFrame before cleaning
        if data_store.raw_df is None:
            data_store.raw_df = df.copy()
            data_store.original_row_count = len(df)  # Store the original row count

        # Initialize variables to track cleaning stats
        rows_dropped_nan = 0
        rows_dropped_blank = 0
        duplicates_removed = 0
        columns_dropped = 0

        # Step 1: Drop unnecessary columns
        columns_to_drop = ["column3", "promo_type_1", "promo_bin_1", "promo_type_2", "promo_bin_2", "promo_discount_2"]
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        df.drop(columns=existing_columns, inplace=True)
        columns_dropped = len(existing_columns)

        # Step 2: Drop blank rows (all values are empty or whitespace)
        initial_rows = len(df)
        blank_rows = df.apply(lambda row: all(pd.isna(val) or str(val).strip() == "" for val in row), axis=1)
        df = df[~blank_rows]
        rows_dropped_blank = initial_rows - len(df)
        logger.info(f"Dropped {rows_dropped_blank} blank rows")

        # Step 3: Drop rows with missing values (NaN)
        initial_rows = len(df)
        df.dropna(inplace=True)
        rows_dropped_nan += initial_rows - len(df)
        logger.info(f"Dropped {rows_dropped_nan} rows with missing values")

        # Step 4: Remove duplicates based on "order_id (unique)"
        if "order_id (unique)" in df.columns:
            initial_rows = len(df)
            df.drop_duplicates(subset="order_id (unique)", keep="first", inplace=True)
            duplicates_removed = initial_rows - len(df)
            logger.info(f"Removed duplicates: {duplicates_removed} rows removed")
        else:
            logger.info("Column 'order_id (unique)' not found, skipping duplicate removal")

        # Step 5: Filter rows to remove NaN in specific columns
        if "order_id (unique)" in df.columns and "price" in df.columns:
            initial_rows = len(df)
            filter_data = df[df["order_id (unique)"].notna() & df["price"].notna()]
            rows_dropped_nan += initial_rows - len(filter_data)
            df = filter_data
        else:
            logger.info("Columns 'order_id (unique)' or 'price' not found, skipping NaN filtering")

        # Step 6: Extract numbers from the "sales" column (handle <number> sales format)
        if "sales" in df.columns:
            if df["sales"].dtype != "object":
                df["sales"] = df["sales"].astype(str)
            df["sales"] = df["sales"].str.extract(r"(\d+)")
        else:
            logger.info("Column 'sales' not found, skipping number extraction")

        # Step 7: Convert "sales" and "revenue" columns to numeric types
        if "sales" in df.columns:
            df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
        else:
            logger.info("Column 'sales' not found, skipping numeric conversion")

        if "revenue" in df.columns:
            df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
        else:
            logger.info("Column 'revenue' not found, skipping numeric conversion")

        # Step 8: Convert "order_date_2" to datetime
        if "order_date_2" in df.columns:
            df["order_date_2"] = pd.to_datetime(df["order_date_2"], errors="coerce")
        else:
            logger.info("Column 'order_date_2' not found, skipping datetime conversion")

        # Compile cleaning messages
        cleaning_messages = []
        if rows_dropped_nan > 0:
            cleaning_messages.append(f"Dropped {rows_dropped_nan} rows with NaN values")
        if duplicates_removed > 0:
            cleaning_messages.append(f"Removed {duplicates_removed} duplicate rows")
        if rows_dropped_blank > 0:
            cleaning_messages.append(f"Dropped {rows_dropped_blank} blank rows")
        if columns_dropped > 0:
            cleaning_messages.append(f"Dropped {columns_dropped} columns: {', '.join(existing_columns)}")

        if not cleaning_messages:
            cleaning_messages.append("No NaN, duplicates, blank rows, or specified columns to drop")

        return df, cleaning_messages
    except Exception as e:
        logger.error(f"Data cleaning error: {e}")
        return None, [f"Error during cleaning: {str(e)}"]

# File parsing function (Improved error handling)
def parse_contents(contents, filename):
    try:
        logger.info(f"Parsing file: {filename}")
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        encoding = chardet.detect(decoded)['encoding'] or 'utf-8'
        logger.info(f"Detected encoding: {encoding}")
        if filename.endswith('.csv'):
            # Limit the number of rows to reduce memory usage
            df = pd.read_csv(io.StringIO(decoded.decode(encoding)), low_memory=False, engine='c', nrows=1000)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')
        else:
            raise ValueError("Unsupported file format")
        
        if df.empty:
            logger.error("Uploaded file is empty")
            return None, ["Uploaded file is empty"]
        
        df, cleaning_messages = clean_data(df)
        if df is None:
            logger.error("Failed to clean data")
            return None, cleaning_messages
        
        logger.info(f"File parsed successfully: {df.shape}")
        return df, cleaning_messages
    except UnicodeDecodeError as e:
        logger.error(f"UnicodeDecodeError: {e}")
        return None, [f"Error decoding file: {str(e)}. Try saving the file with UTF-8 encoding."]
    except pd.errors.ParserError as e:
        logger.error(f"ParserError: {e}")
        return None, [f"Error parsing CSV file: {str(e)}. Check the file format."]
    except Exception as e:
        logger.error(f"File parsing error: {e}")
        return None, [f"Error parsing file: {str(e)}"]

# Chart generation functions
def generate_main_chart(df, selected_column, selected_y_axis, chart_type, dark_mode, selected_category):
    try:
        if not selected_column or not selected_y_axis:
            return px.scatter(title="Please select X and Y axes")
        
        # Sample the dataset to reduce memory usage
        if len(df) > 1000:
            df = df.sample(1000, random_state=42)
        
        template = 'plotly_dark' if dark_mode else 'plotly'
        if chart_type == 'bar':
            fig = px.bar(df, x=selected_column, y=selected_y_axis, template=template)
            if selected_category:
                fig.update_traces(marker=dict(color=['#00d4ff' if x == selected_category else '#636efa' 
                                                  for x in df[selected_column]]))
        elif chart_type == 'pie':
            fig = px.pie(df, names=selected_column, values=selected_y_axis, template=template)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=selected_column, y=selected_y_axis, template=template)
            if selected_category:
                fig.update_traces(marker=dict(size=[15 if x == selected_category else 8 
                                                 for x in df[selected_column]]))
        elif chart_type == 'line':
            fig = px.line(df, x=selected_column, y=selected_y_axis, template=template)
            if selected_category:
                fig.add_scatter(x=[selected_category], y=[df[df[selected_column] == selected_category][selected_y_axis].iloc[0]],
                              mode='markers', marker=dict(size=15, color='#00d4ff'), showlegend=False)
        elif chart_type == 'heatmap':
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numerical_cols) >= 2:
                corr_matrix = df[numerical_cols].corr()
                fig = ff.create_annotated_heatmap(z=corr_matrix.values, x=numerical_cols, y=numerical_cols,
                                                colorscale='Plasma' if dark_mode else 'Viridis')
            else:
                fig = px.scatter(title="Not enough numerical columns for heatmap")
        elif chart_type == 'box':
            fig = px.box(df, x=selected_column, y=selected_y_axis, template=template)
        elif chart_type == 'cluster':
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numerical_cols) >= 2:
                kmeans = KMeans(n_clusters=3, n_init=10)
                features = df[numerical_cols].dropna()
                df.loc[features.index, 'cluster'] = kmeans.fit_predict(features)
                fig = px.scatter(df, x=numerical_cols[0], y=numerical_cols[1], color='cluster', template=template)
            else:
                fig = px.scatter(title="Not enough numerical columns for clustering")
        elif chart_type == 'regression':
            if (df[selected_column].dtype in ['int64', 'float64'] and 
                df[selected_y_axis].dtype in ['int64', 'float64']):
                X = df[[selected_column]].dropna()
                y = df.loc[X.index, selected_y_axis]
                model = LinearRegression()
                model.fit(X, y)
                df.loc[X.index, 'predicted'] = model.predict(X)
                fig = px.scatter(df, x=selected_column, y=selected_y_axis, trendline="ols", template=template)
                if selected_category:
                    fig.update_traces(marker=dict(size=[15 if x == selected_category else 8 
                                                     for x in df[selected_column]]))
            else:
                fig = px.scatter(title="Regression requires numerical X and Y axes")
        else:
            fig = px.scatter(title="Unsupported chart type")
        return fig
    except Exception as e:
        logger.error(f"Main chart generation error: {e}")
        return px.scatter(title=f"Error: {str(e)}")

def generate_pie_chart(df, selected_column, selected_y_axis, dark_mode, selected_category):
    try:
        if not selected_column or not selected_y_axis:
            return px.scatter(title="No pie chart generated")
        # Sample the dataset to reduce memory usage
        if len(df) > 1000:
            df = df.sample(1000, random_state=42)
        fig = px.pie(df, names=selected_column, values=selected_y_axis, 
                    template='plotly_dark' if dark_mode else 'plotly',
                    title=f"Pie: {selected_column} Distribution")
        if selected_category:
            explode = [0.1 if x == selected_category else 0 for x in df[selected_column].unique()]
            fig.update_traces(pull=explode, marker=dict(colors=['#00d4ff' if x == selected_category else '#636efa' 
                                                      for x in df[selected_column].unique()]))
        return fig
    except Exception as e:
        logger.error(f"Pie chart generation error: {e}")
        return px.scatter(title=f"Error: {str(e)}")

def generate_line_chart(df, selected_column, selected_y_axis, dark_mode, selected_category):
    try:
        if not selected_column or not selected_y_axis:
            return px.scatter(title="No line chart generated")
        # Sample the dataset to reduce memory usage
        if len(df) > 1000:
            df = df.sample(1000, random_state=42)
        fig = px.line(df, x=selected_column, y=selected_y_axis,
                     template='plotly_dark' if dark_mode else 'plotly',
                     title=f"Line: {selected_column} vs {selected_y_axis}")
        if selected_category:
            fig.add_scatter(x=[selected_category], y=[df[df[selected_column] == selected_category][selected_y_axis].iloc[0]],
                          mode='markers', marker=dict(size=15, color='#00d4ff'), showlegend=False)
        return fig
    except Exception as e:
        logger.error(f"Line chart generation error: {e}")
        return px.scatter(title=f"Error: {str(e)}")

# Analysis Functions
def generate_correlation_heatmap(df, dark_mode):
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numerical_cols) < 2:
        return px.scatter(title="Not enough numerical columns for correlation analysis")
    # Sample the dataset to reduce memory usage
    if len(df) > 1000:
        df = df.sample(1000, random_state=42)
    corr_matrix = df[numerical_cols].corr()
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=numerical_cols,
        y=numerical_cols,
        colorscale='Plasma' if dark_mode else 'Viridis',
        showscale=True
    )
    fig.update_layout(
        title="Correlation Heatmap",
        template='plotly_dark' if dark_mode else 'plotly'
    )
    return fig

def detect_outliers(df, column):
    if column not in df.columns or df[column].dtype not in ['int64', 'float64']:
        return [], "Select a numerical column for outlier detection"
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    if outliers.empty:
        return [], "No outliers detected"
    outlier_options = [{"label": f"Row {idx}: {val}", "value": f"Row {idx}: {val}"} for idx, val in outliers.items()]
    return outlier_options, None

def generate_trend_chart(df, date_column, y_axis, dark_mode):
    if not date_column or not y_axis or date_column not in df.columns or y_axis not in df.columns:
        return px.scatter(title="Select a date column and Y-axis for trend analysis")
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    # Sample the dataset to reduce memory usage
    if len(df) > 1000:
        df = df.sample(1000, random_state=42)
    trend_data = df.groupby(date_column)[y_axis].sum().reset_index()
    fig = px.line(trend_data, x=date_column, y=y_axis, template='plotly_dark' if dark_mode else 'plotly')
    fig.update_layout(title=f"Trend of {y_axis} Over Time")
    return fig

# Data Science Functions (Optimized for memory usage)
def generate_forecast(df, date_column, y_axis, periods=15, dark_mode=True):
    if not date_column or not y_axis or date_column not in df.columns or y_axis not in df.columns:
        return px.scatter(title="Select a date column and Y-axis for forecasting")
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    # Sample the dataset to reduce memory usage
    if len(df) > 1000:
        df = df.sample(1000, random_state=42)
    forecast_data = df[[date_column, y_axis]].rename(columns={date_column: 'ds', y_axis: 'y'})
    forecast_data = forecast_data.dropna()
    if len(forecast_data) < 2:
        return px.scatter(title="Not enough data for forecasting")
    # Adjust seasonality based on dataset size
    model = Prophet(
        yearly_seasonality=len(forecast_data) > 365,
        weekly_seasonality=len(forecast_data) > 90,
        daily_seasonality=len(forecast_data) > 30
    )
    model.fit(forecast_data)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['y'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='lines', line=dict(color='rgba(0,0,0,0)'), name='Confidence Interval'))
    fig.update_layout(
        title=f"Forecast of {y_axis} for Next {periods} Days",
        template='plotly_dark' if dark_mode else 'plotly'
    )
    return fig

def generate_clustering_report(df):
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numerical_cols) < 2:
        return "Not enough numerical columns for clustering report"
    # Sample the dataset to reduce memory usage
    if len(df) > 1000:
        df = df.sample(1000, random_state=42)
    kmeans = KMeans(n_clusters=3, n_init=10)
    features = df[numerical_cols].dropna()
    df.loc[features.index, 'cluster'] = kmeans.fit_predict(features)
    cluster_summary = df.groupby('cluster')[numerical_cols].mean().reset_index()
    report = []
    for cluster in cluster_summary['cluster']:
        cluster_data = cluster_summary[cluster_summary['cluster'] == cluster]
        report.append(html.P(f"Cluster {int(cluster)}:", className="text-light"))
        report.append(html.Ul([
            html.Li(f"{col}: {cluster_data[col].iloc[0]:.2f}") for col in numerical_cols
        ], className="text-light"))
    return report

def generate_feature_importance(df, target_column, dark_mode):
    if target_column not in df.columns or df[target_column].dtype not in ['int64', 'float64']:
        return px.scatter(title="Select a numerical target column for feature importance")
    features = df.select_dtypes(include=['number']).columns.tolist()
    features = [col for col in features if col != target_column]
    if not features:
        return px.scatter(title="Not enough numerical features for analysis")
    # Sample the dataset to reduce memory usage
    if len(df) > 1000:
        df = df.sample(1000, random_state=42)
    X = df[features].fillna(0)
    y = df[target_column].fillna(0)
    # Reduce the number of trees to lower memory usage
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    fig = px.bar(x=importances, y=features, orientation='h', template='plotly_dark' if dark_mode else 'plotly')
    fig.update_layout(title=f"Feature Importance for {target_column}", xaxis_title="Importance", yaxis_title="Feature")
    return fig

# Sidebar content
sidebar = dbc.Col([
    dbc.Card([
        dbc.CardBody([
            html.H4("Controls", className="card-title text-light"),
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload File 📁', className="btn btn-outline-cyan btn-block mb-3"),
                multiple=False,
                accept=".csv,.xlsx"
            ),
            html.Div(id='cleaning-message', className="mb-3 text-center text-light"),
            html.Div(id='upload-message', className="mb-3 text-center text-light"),
            dcc.Dropdown(id='column-filter', placeholder="Select a column", value='order_id', className="mb-3 dropdown-purple"),
            dcc.Dropdown(id='value-filter', placeholder="Select values", value=[], multi=True, className="mb-3 dropdown-purple"),
            dcc.Dropdown(id='y-axis-dropdown', placeholder="Select Y-axis", value='store_id', className="mb-3 dropdown-purple"),
            dcc.Dropdown(id='chart-type', options=[
                {'label': 'Bar Chart', 'value': 'bar'},
                {'label': 'Pie Chart', 'value': 'pie'},
                {'label': 'Scatter Plot', 'value': 'scatter'},
                {'label': 'Line Chart', 'value': 'line'},
                {'label': 'Heatmap', 'value': 'heatmap'},
                {'label': 'Box Plot', 'value': 'box'},
                {'label': 'Clustering', 'value': 'cluster'},
                {'label': 'Regression', 'value': 'regression'}
            ], value='bar', className="mb-3 dropdown-purple"),
            dcc.Dropdown(id='date-column', placeholder="Select Date Column", className="mb-3 dropdown-purple"),
            dcc.Dropdown(id='target-column', placeholder="Select Target Column for Feature Importance", className="mb-3 dropdown-purple"),
            dbc.Switch(id='dark-mode-toggle', label='Dark Mode', value=True, className="mb-3 text-light"),
            html.Button("Download Data 📥", id="download-button", className="btn btn-outline-cyan btn-block mb-3"),
            html.Button("Clear Selection", id="clear-selection", className="btn btn-outline-red btn-block mb-3"),
            html.H5("What-If Analysis", className="card-title text-light mt-3"),
            html.Label("Adjust Price (% Change)", className="text-light"),
            dcc.Slider(id='price-adjust', min=-50, max=50, step=5, value=0, marks={i: f"{i}%" for i in range(-50, 51, 25)}),
            html.Label("Adjust Sales (% Change)", className="text-light mt-2"),
            dcc.Slider(id='sales-adjust', min=-50, max=50, step=5, value=0, marks={i: f"{i}%" for i in range(-50, 51, 25)}),
            html.H5("Insights", className="card-title text-light mt-3"),
            html.Div(id='insights-panel', style={'maxHeight': '200px', 'overflowY': 'auto'}),
            html.H5("Summary Statistics", className="card-title text-light mt-3"),
            html.Div(id='summary-stats'),
            html.H5("Outlier Detection", className="card-title text-light mt-3"),
            dcc.Dropdown(id='outlier-detection', placeholder="Select an outlier", options=[], className="mb-3 dropdown-purple"),
            html.H5("Clustering Report", className="card-title text-light mt-3"),
            html.Div(id='clustering-report')
        ])
    ], className="bg-card-dark border-0 shadow-sm")
], width=3, className="sidebar collapse show", id="sidebar")

# Main content (Added loading spinners)
main_content = dbc.Col([
    dbc.NavbarSimple(brand="📊 Advanced Data Analysis Dashboard", color="dark", dark=True, className="mb-4 navbar-dark"),
    html.Div(id='row-comparison', className="mb-3 text-center text-light"),
    dbc.Card([
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab(label="Main Chart", tab_id="main-tab", children=[
                    dcc.Loading(
                        id="loading-main-chart",
                        type="default",
                        children=dcc.Graph(id='main-chart', style={'height': '400px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']})
                    )
                ]),
                dbc.Tab(label="Pie Chart", tab_id="pie-tab", children=[
                    dcc.Loading(
                        id="loading-pie-chart",
                        type="default",
                        children=dcc.Graph(id='pie-chart', style={'height': '400px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']})
                    )
                ]),
                dbc.Tab(label="Line Chart", tab_id="line-tab", children=[
                    dcc.Loading(
                        id="loading-line-chart",
                        type="default",
                        children=dcc.Graph(id='line-chart', style={'height': '400px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']})
                    )
                ]),
                dbc.Tab(label="Correlation", tab_id="correlation-tab", children=[
                    dcc.Loading(
                        id="loading-correlation-heatmap",
                        type="default",
                        children=dcc.Graph(id='correlation-heatmap', style={'height': '400px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']})
                    )
                ]),
                dbc.Tab(label="Trends", tab_id="trends-tab", children=[
                    dcc.Loading(
                        id="loading-trend-chart",
                        type="default",
                        children=dcc.Graph(id='trend-chart', style={'height': '400px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']})
                    )
                ]),
                dbc.Tab(label="Forecast", tab_id="forecast-tab", children=[
                    dcc.Loading(
                        id="loading-forecast-chart",
                        type="default",
                        children=dcc.Graph(id='forecast-chart', style={'height': '400px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']})
                    )
                ]),
                dbc.Tab(label="Feature Importance", tab_id="feature-importance-tab", children=[
                    dcc.Loading(
                        id="loading-feature-importance-chart",
                        type="default",
                        children=dcc.Graph(id='feature-importance-chart', style={'height': '400px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']})
                    )
                ]),
                dbc.Tab(label="All Charts", tab_id="all-tab", children=[
                    html.Button("Download All Charts 📥", id="download-all-charts", className="btn btn-outline-cyan btn-block mb-3"),
                    dcc.Download(id="download-all-charts-zip"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='main-chart-all', style={'height': '300px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']}), width=4),
                        dbc.Col(dcc.Graph(id='pie-chart-all', style={'height': '300px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']}), width=4),
                        dbc.Col(dcc.Graph(id='line-chart-all', style={'height': '300px'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']}), width=4)
                    ])
                ])
            ], id="chart-tabs", active_tab="main-tab", className="tabs-custom")
        ])
    ], className="bg-card-dark border-0 shadow-sm mb-4"),
    dbc.Card([
        dbc.CardBody([
            html.H5("Data Table", className="card-title text-light"),
            html.Div(id='data-table-debug'),
            html.Div(id='data-table-error', style={'color': 'red'}),
            dash_table.DataTable(
                id='data-table',
                columns=[],
                data=[],
                style_table={'overflowX': 'auto', 'height': '400px'},
                style_cell={'textAlign': 'left', 'backgroundColor': '#2e3b55', 'color': 'white'},
                style_header={'backgroundColor': '#1f2a44', 'color': 'white', 'fontWeight': 'bold'},
                style_data={'border': '1px solid #ffffff20'},
                page_action='custom',
                page_current=0,
                page_size=10,
            )
        ])
    ], className="bg-card-dark border-0 shadow-sm"),
    dcc.Download(id="download-dataframe-csv"),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Enlarged Chart"), className="bg-card-dark text-light"),
        dbc.ModalBody(dcc.Graph(id='enlarged-chart', style={'height': '70vh'}, config={'displaylogo': False, 'modeBarButtonsToAdd': ['downloadImage']}), className="bg-card-dark"),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="btn btn-outline-cyan")
        )
    ], id="chart-modal", size="xl", is_open=False, backdrop=True, className="modal-dark"),
    dcc.Store(id='selected-category')
], width=9)

# Layout
app.layout = dbc.Container([
    dbc.Row([
        sidebar,
        main_content
    ], className="min-vh-100")
], fluid=True, className="bg-gradient-dark")

# Main callback (Updated to preserve data_store.df)
@app.callback(
    [Output('column-filter', 'options'),
     Output('value-filter', 'options'),
     Output('y-axis-dropdown', 'options'),
     Output('date-column', 'options'),
     Output('target-column', 'options'),
     Output('main-chart', 'figure'),
     Output('pie-chart', 'figure'),
     Output('line-chart', 'figure'),
     Output('correlation-heatmap', 'figure'),
     Output('trend-chart', 'figure'),
     Output('forecast-chart', 'figure'),
     Output('feature-importance-chart', 'figure'),
     Output('insights-panel', 'children'),
     Output('data-table', 'data'),
     Output('data-table', 'columns'),
     Output('upload-message', 'children'),
     Output('summary-stats', 'children'),
     Output('selected-category', 'data'),
     Output('data-table-debug', 'children'),
     Output('data-table-error', 'children'),
     Output('cleaning-message', 'children'),
     Output('row-comparison', 'children'),
     Output('outlier-detection', 'options'),
     Output('outlier-detection', 'placeholder'),
     Output('clustering-report', 'children')],
    [Input('upload-data', 'contents'),
     Input('column-filter', 'value'),
     Input('value-filter', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('chart-type', 'value'),
     Input('dark-mode-toggle', 'value'),
     Input('main-chart', 'clickData'),
     Input('clear-selection', 'n_clicks'),
     Input('data-table', 'page_current'),
     Input('data-table', 'page_size'),
     Input('date-column', 'value'),
     Input('target-column', 'value'),
     Input('price-adjust', 'value'),
     Input('sales-adjust', 'value')],
    [State('upload-data', 'filename')]
)
def update_dashboard(contents, selected_column, selected_values, selected_y_axis, chart_type, dark_mode, click_data, clear_clicks, page_current, page_size, date_column, target_column, price_adjust, sales_adjust, filename):
    try:
        logger.info("update_dashboard callback triggered")
        logger.info(f"Contents: {contents is not None}, Filename: {filename}")

        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        # Default outputs
        default_fig = px.scatter(title="Please upload a file")
        default_outputs = [[] for _ in range(5)] + [default_fig] * 7 + [[] for _ in range(3)] + ["Please upload a file", "No data available", None, "No data", "", "No cleaning performed", "", [], "No outliers detected", "No clustering performed"]

        if not contents and data_store.df is None:
            logger.info("No contents and no stored data, returning default outputs")
            return default_outputs

        if triggered_id == 'upload-data':
            if contents:
                logger.info("Processing new upload")
                df, cleaning_messages = parse_contents(contents, filename)
                if df is not None:
                    data_store.df = df
                else:
                    logger.error("Failed to parse contents")
                    default_fig = px.scatter(title="Error loading file")
                    return [[] for _ in range(5)] + [default_fig] * 7 + [[] for _ in range(3)] + ["❌ Error loading file", "Error loading data", None, "Error", "Failed to load data table", html.Ul([html.Li(msg) for msg in cleaning_messages]), "", [], "Error loading data", "Error loading data"]
            else:
                logger.error("No contents provided for new upload")
                return default_outputs
        else:
            logger.info("Using stored DataFrame")
            df = data_store.df
            cleaning_messages = ["Using previously cleaned data"]

        if df is None:
            logger.error("DataFrame is None")
            default_fig = px.scatter(title="No data available")
            return [[] for _ in range(5)] + [default_fig] * 7 + [[] for _ in range(3)] + ["No data available", "No data available", None, "No data", "No data available", html.Ul([html.Li(msg) for msg in cleaning_messages]), "", [], "No data available", "No data available"]

        logger.info(f"DataFrame shape after loading: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")

        row_comparison_message = ""
        if filename == "up sales 2017.csv":
            try:
                cleaned_data = pd.read_csv("cleaned_data 2017.csv")
                up_sales_rows = data_store.original_row_count
                cleaned_data_rows = len(cleaned_data)
                row_comparison_message = (
                    f"Rows in 'up sales 2017.csv': {up_sales_rows} | "
                    f"Rows in 'cleaned_data 2017.csv': {cleaned_data_rows} | "
                    f"Difference: {abs(up_sales_rows - cleaned_data_rows)}"
                )
            except FileNotFoundError:
                row_comparison_message = "Error: 'cleaned_data 2017.csv' not found for row comparison"
            except Exception as e:
                row_comparison_message = f"Error comparing rows: {str(e)}"

        logger.info("Selecting categorical and numerical columns")
        categorical_cols = df.select_dtypes(include=['object', 'datetime']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

        selected_column = selected_column or (categorical_cols[0] if categorical_cols else None)
        selected_y_axis = selected_y_axis or (numerical_cols[0] if numerical_cols else None)
        chart_type = chart_type or 'bar'

        logger.info("Generating value options for dropdown")
        value_options = ([{'label': str(val), 'value': val} 
                         for val in df[selected_column].dropna().unique()] 
                         if selected_column in df.columns else [])
        
        logger.info("Filtering DataFrame")
        df_filtered = df[df[selected_column].isin(selected_values)] if selected_column and selected_values else df.copy()
        data_store.filtered_df = df_filtered
        logger.info(f"Filtered DataFrame shape: {df_filtered.shape}")
        logger.info(f"Filtered DataFrame columns: {df_filtered.columns.tolist()}")

        if 'price' in df_filtered.columns:
            df_filtered['price'] = df_filtered['price'] * (1 + price_adjust / 100)
        if 'sales' in df_filtered.columns:
            df_filtered['sales'] = df_filtered['sales'] * (1 + sales_adjust / 100)
        if 'price' in df_filtered.columns and 'sales' in df_filtered.columns:
            df_filtered['revenue'] = df_filtered['price'] * df_filtered['sales']

        selected_category = None
        if triggered_id == 'main-chart' and click_data and chart_type in ['bar', 'scatter', 'line', 'regression']:
            logger.info("Handling main chart click data")
            selected_category = click_data['points'][0].get('x') or click_data['points'][0].get('label')
        elif triggered_id == 'clear-selection' and clear_clicks:
            logger.info("Clearing selection")
            selected_category = None

        logger.info("Generating main chart")
        main_fig = generate_main_chart(df_filtered, selected_column, selected_y_axis, chart_type, dark_mode, selected_category)
        logger.info("Generating pie chart")
        pie_fig = generate_pie_chart(df_filtered, selected_column, selected_y_axis, dark_mode, selected_category)
        logger.info("Generating line chart")
        line_fig = generate_line_chart(df_filtered, selected_column, selected_y_axis, dark_mode, selected_category)
        logger.info("Generating correlation heatmap")
        correlation_fig = generate_correlation_heatmap(df_filtered, dark_mode)
        logger.info("Generating trend chart")
        trend_fig = generate_trend_chart(df_filtered, date_column, selected_y_axis, dark_mode)
        logger.info("Generating forecast chart")
        forecast_fig = generate_forecast(df_filtered, date_column, selected_y_axis, periods=15, dark_mode=dark_mode)
        logger.info("Generating feature importance chart")
        feature_importance_fig = generate_feature_importance(df_filtered, target_column, dark_mode)

        logger.info("Generating insights")
        insights = [html.P(f"🔹 {col}: {df[col].nunique()} unique", className="text-light") 
                    for col in df.columns]

        logger.info("Generating summary statistics")
        summary_stats = "Select a Y-axis for statistics"
        if selected_y_axis and selected_y_axis in numerical_cols:
            stats = df_filtered[selected_y_axis].describe()
            summary_stats = html.Ul([
                html.Li(f"Mean: {stats['mean']:.2f}", className="text-light"),
                html.Li(f"Median: {stats['50%']:.2f}", className="text-light"),
                html.Li(f"Min: {stats['min']:.2f}", className="text-light"),
                html.Li(f"Max: {stats['max']:.2f}", className="text-light"),
                html.Li(f"Count: {int(stats['count'])}", className="text-light"),
                html.Li(f"Std Dev: {stats['std']:.2f}", className="text-light")
            ], className="list-unstyled")

        logger.info("Detecting outliers")
        outlier_options, outlier_message = detect_outliers(df_filtered, selected_y_axis)

        logger.info("Generating clustering report")
        clustering_report = generate_clustering_report(df_filtered) if chart_type == 'cluster' else "Select 'Clustering' chart type to view report"

        logger.info("Converting DataFrame to string for DataTable")
        df_filtered = df_filtered.astype(str)
        logger.info("Preparing row data with pagination")
        start_idx = page_current * page_size
        end_idx = start_idx + page_size
        row_data = df_filtered.iloc[start_idx:end_idx].to_dict('records')
        logger.info(f"Paginated row data prepared: {len(row_data)} rows (page {page_current + 1})")
        logger.info(f"Sample row data: {row_data[:2] if row_data else 'Empty'}")

        logger.info("Generating column definitions")
        if df_filtered.empty:
            column_defs = [{"name": "Message", "id": "message"}]
            row_data = [{"message": "No data available after filtering"}]
            logger.warning("DataFrame is empty, displaying placeholder message in table")
        else:
            column_defs = [{"name": col.replace('_', ' ').title(), "id": col} for col in df_filtered.columns]
        logger.info(f"Column definitions generated: {column_defs}")

        debug_info = f"Rows: {len(row_data)}, Columns: {len(column_defs)}"

        logger.info("Returning callback outputs")
        return (
            [{'label': col, 'value': col} for col in categorical_cols],
            value_options,
            [{'label': col, 'value': col} for col in numerical_cols],
            [{'label': col, 'value': col} for col in datetime_cols],
            [{'label': col, 'value': col} for col in numerical_cols],
            main_fig,
            pie_fig,
            line_fig,
            correlation_fig,
            trend_fig,
            forecast_fig,
            feature_importance_fig,
            insights,
            row_data,
            column_defs,
            f"✅ Uploaded {filename} ({len(df)} rows)" if contents else f"Data loaded ({len(df)} rows)",
            summary_stats,
            selected_category,
            debug_info,
            "",
            html.Ul([html.Li(msg) for msg in cleaning_messages]),
            row_comparison_message,
            outlier_options,
            outlier_message if outlier_message else "Select an outlier",
            clustering_report
        )
    except Exception as e:
        logger.error(f"Dashboard update error: {str(e)}", exc_info=True)
        default_fig = px.scatter(title=f"Dashboard error: {str(e)}")
        return [[] for _ in range(5)] + [default_fig] * 7 + [[] for _ in range(3)] + [f"❌ Error: {str(e)}", "Error loading data", None, "Error", f"Failed to load data table: {str(e)}", "Error during update", "", [], "Error loading data", "Error loading data"]

# Separate callback for "All Charts" tab
@app.callback(
    [Output('main-chart-all', 'figure'),
     Output('pie-chart-all', 'figure'),
     Output('line-chart-all', 'figure')],
    [Input('chart-tabs', 'active_tab'),
     Input('column-filter', 'value'),
     Input('value-filter', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('chart-type', 'value'),
     Input('dark-mode-toggle', 'value')]
)
def update_all_charts_tab(active_tab, selected_column, selected_values, selected_y_axis, chart_type, dark_mode):
    try:
        if active_tab != 'all-tab':
            return [px.scatter()] * 3

        if data_store.df is None:
            return [px.scatter(title="No data available")] * 3

        df = data_store.df
        df_filtered = df[df[selected_column].isin(selected_values)] if selected_column and selected_values else df.copy()

        main_fig_all = generate_main_chart(df_filtered, selected_column, selected_y_axis, chart_type, dark_mode, None)
        pie_fig_all = generate_pie_chart(df_filtered, selected_column, selected_y_axis, dark_mode, None)
        line_fig_all = generate_line_chart(df_filtered, selected_column, selected_y_axis, dark_mode, None)

        return main_fig_all, pie_fig_all, line_fig_all
    except Exception as e:
        logger.error(f"All charts update error: {str(e)}")
        return [px.scatter(title=f"Error: {str(e)}")] * 3

# Callback to handle "Download All Charts" button
@app.callback(
    Output("download-all-charts-zip", "data"),
    Input("download-all-charts", "n_clicks"),
    State('main-chart-all', 'figure'),
    State('pie-chart-all', 'figure'),
    State('line-chart-all', 'figure'),
    prevent_initial_call=True
)
def download_all_charts(n_clicks, main_fig, pie_fig, line_fig):
    if not n_clicks:
        return None

    if not main_fig or not pie_fig or not line_fig:
        logger.error("One or more charts are not available for download")
        return None

    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for fig, name in [(main_fig, 'main_chart.png'), (pie_fig, 'pie_chart.png'), (line_fig, 'line_chart.png')]:
                if fig:
                    fig_obj = go.Figure(fig)
                    img_buffer = io.BytesIO()
                    fig_obj.write_image(img_buffer, format='png', engine="kaleido", width=800, height=600)
                    img_buffer.seek(0)
                    zip_file.writestr(name, img_buffer.getvalue())

        zip_buffer.seek(0)
        return dict(
            content=base64.b64encode(zip_buffer.getvalue()).decode(),
            filename="all_charts.zip",
            type="application/zip",
            base64=True
        )
    except Exception as e:
        logger.error(f"Error creating ZIP file: {str(e)}")
        return None

# Modal callback for enlarged chart
@app.callback(
    [Output('chart-modal', 'is_open'),
     Output('enlarged-chart', 'figure')],
    [Input('main-chart-all', 'clickData'),
     Input('pie-chart-all', 'clickData'),
     Input('line-chart-all', 'clickData'),
     Input('close-modal', 'n_clicks')],
    [State('main-chart-all', 'figure'),
     State('pie-chart-all', 'figure'),
     State('line-chart-all', 'figure'),
     State('chart-modal', 'is_open')]
)
def toggle_modal(main_click, pie_click, line_click, close_click, main_fig, pie_fig, line_fig, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, {}
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'close-modal' and close_click:
        return False, {}
    
    if triggered_id == 'main-chart-all' and main_click:
        return True, main_fig
    elif triggered_id == 'pie-chart-all' and pie_click:
        return True, pie_fig
    elif triggered_id == 'line-chart-all' and line_click:
        return True, line_fig
    
    return is_open, {}

# Download callback for filtered data
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_data(n_clicks):
    if data_store.filtered_df is not None:
        return dcc.send_data_frame(data_store.filtered_df.to_csv, "filtered_dashboard_data.csv")
    elif data_store.df is not None:
        return dcc.send_data_frame(data_store.df.to_csv, "dashboard_data.csv")
    return None

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .bg-gradient-dark {
                background: linear-gradient(135deg, #1f2a44 0%, #2e3b55 100%);
            }
            .bg-card-dark {
                background-color: #2e3b55;
            }
            .navbar-dark {
                background-color: #1f2a44 !important;
            }
            .sidebar {
                transition: all 0.3s ease;
            }
            .sidebar.collapsed {
                margin-left: -25%;
            }
            .card {
                border-radius: 12px;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                border: 1px solid #ffffff10;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            }
            .btn-outline-cyan {
                border-color: #00d4ff;
                color: #00d4ff;
                transition: background-color 0.2s ease;
            }
            .btn-outline-cyan:hover {
                background-color: #00d4ff20;
                color: #00d4ff;
            }
            .btn-outline-red {
                border-color: #ff4d4f;
                color: #ff4d4f;
                transition: background-color 0.2s ease;
            }
            .btn-outline-red:hover {
                background-color: #ff4d4f20;
                color: #ff4d4f;
            }
            .tabs-custom .nav-link {
                color: #ffffff80;
                border-radius: 8px 8px 0 0;
                transition: background-color 0.2s ease;
            }
            .tabs-custom .nav-link.active {
                color: #ffffff;
                background-color: #2e3b55;
                border-bottom: 3px solid #00d4ff;
            }
            .tabs-custom .nav-link:hover {
                color: #ffffff;
                background-color: #ffffff10;
            }
            .dropdown-purple .Select-control {
                background-color: #9b59b6 !important;
                border-color: #9b59b6 !important;
                transition: opacity 0.2s ease;
            }
            .Select-menu-outer {
                background-color: #2e3b55 !important;
                border-color: #ffffff20 !important;
            }
            .Select-option {
                color: #ffffff !important;
                transition: background-color 0.1s ease;
            }
            .Select-option.is-focused {
                background-color: #00d4ff20 !important;
            }
            .text-light {
                color: #e6e6e6 !important;
            }
            .modal-dark .modal-content {
                background-color: #2e3b55;
                border: 1px solid #ffffff10;
                animation: fadeIn 0.3s ease;
            }
            .modal-dark .modal-header {
                border-bottom: 1px solid #ffffff20;
            }
            .modal-dark .modal-footer {
                border-top: 1px solid #ffffff20;
            }
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
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

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050)