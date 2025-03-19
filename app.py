import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import base64
import os
import io
import chardet
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import plotly.figure_factory as ff
import dash_ag_grid as dag
from datetime import datetime
import logging
from scipy.stats import ttest_ind
from prophet import Prophet
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash App with PWA support
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="📊 Advanced Data Analysis Dashboard",
    assets_folder='static'  # Static files for PWA
)

# Enable Flask to serve static files
app.server.static_folder = 'static'
server = app.server
port = int(os.environ.get("PORT", 8050))

# Global data store
class DataStore:
    def __init__(self):
        self.df = None
        self.last_updated = None
        self.filtered_df = None

data_store = DataStore()

# Data cleaning function
def clean_data(df):
    try:
        df.columns = (df.columns.str.strip()
                     .str.replace(" ", "_", regex=False)
                     .str.replace(r"[^\w\s]", "", regex=True)
                     .str.lower())
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    try:
                        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')
                    except ValueError:
                        pass
        return df.ffill().bfill()
    except Exception as e:
        logger.error(f"Data cleaning error: {e}")
        return None

# File parsing function
def parse_contents(contents, filename):
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
        df = clean_data(df)
        if df is not None:
            data_store.df = df
            data_store.last_updated = datetime.now()
        return df
    except Exception as e:
        logger.error(f"File parsing error: {e}")
        return None

# Data transformation function
def transform_data(df, col, transform_type):
    try:
        if transform_type == 'log':
            return np.log1p(df[col].replace([np.inf, -np.inf], np.nan))
        elif transform_type == 'normalize':
            return (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        elif transform_type == 'none' or transform_type is None:
            return df[col]
        return df[col]
    except Exception as e:
        logger.error(f"Data transformation error: {e}")
        return df[col]

# Outlier detection
def detect_outliers(df, col):
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return df[col].apply(lambda x: 'outlier' if x < lower or x > upper else 'normal')

# Chart generation functions
def generate_main_chart(df, selected_column, selected_y_axis, chart_type, dark_mode, selected_category, show_outliers, cluster_count, transform_type):
    if not selected_column or not selected_y_axis:
        return px.scatter(title="Please select X and Y axes")
    try:
        df_copy = df.copy()
        x_data = transform_data(df_copy, selected_column, transform_type)
        y_data = transform_data(df_copy, selected_y_axis, transform_type)

        if show_outliers:
            df_copy['outlier'] = detect_outliers(df_copy, selected_column)
            color = 'outlier'
        else:
            color = None

        if cluster_count and cluster_count > 0:
            kmeans = KMeans(n_clusters=cluster_count, random_state=0)
            features = df_copy[[selected_column, selected_y_axis]].dropna()
            if len(features) > 0:
                df_copy['cluster'] = kmeans.fit_predict(features)
                color = 'cluster'

        if chart_type == 'scatter':
            fig = px.scatter(df_copy, x=x_data, y=y_data, color=color,
                             title=f"{selected_column} vs {selected_y_axis}")
        elif chart_type == 'bar':
            fig = px.bar(df_copy, x=x_data, y=y_data, color=color,
                         title=f"{selected_column} vs {selected_y_axis}")
        else:  # line
            fig = px.line(df_copy, x=x_data, y=y_data, color=color,
                          title=f"{selected_column} vs {selected_y_axis}")

        if dark_mode:
            fig.update_layout(template="plotly_dark")
        return fig
    except Exception as e:
        logger.error(f"Main chart generation error: {e}")
        return px.scatter(title=f"Error generating chart: {str(e)}")

def generate_pie_chart(df, selected_column, selected_y_axis, dark_mode, selected_category, transform_type):
    if not selected_column or not selected_y_axis:
        return px.scatter(title="No pie chart generated")
    try:
        df_copy = df.copy()
        data = transform_data(df_copy, selected_y_axis, transform_type)
        fig = px.pie(df_copy, names=selected_column, values=data,
                     title=f"Pie Chart of {selected_y_axis} by {selected_column}")
        if dark_mode:
            fig.update_layout(template="plotly_dark")
        return fig
    except Exception as e:
        logger.error(f"Pie chart generation error: {e}")
        return px.scatter(title=f"Error generating pie chart: {str(e)}")

def generate_line_chart(df, selected_column, selected_y_axis, dark_mode, selected_category, transform_type):
    if not selected_column or not selected_y_axis:
        return px.scatter(title="No line chart generated")
    try:
        df_copy = df.copy()
        x_data = transform_data(df_copy, selected_column, transform_type)
        y_data = transform_data(df_copy, selected_y_axis, transform_type)
        fig = px.line(df_copy, x=x_data, y=y_data,
                      title=f"Line Chart of {selected_y_axis} over {selected_column}")
        if dark_mode:
            fig.update_layout(template="plotly_dark")
        return fig
    except Exception as e:
        logger.error(f"Line chart generation error: {e}")
        return px.scatter(title=f"Error generating line chart: {str(e)}")

# Sidebar content
sidebar = dbc.Col([
    dbc.Card([
        dbc.CardBody([
            html.H4("Controls", className="card-title text-light"),
            dcc.Upload(id='upload-data', children=html.Button('Upload File 📁', className="btn btn-outline-cyan btn-block mb-3"), multiple=False, accept=".csv,.xlsx"),
            dcc.Dropdown(id='column-filter', options=[], placeholder="Select a column", className="dropdown-blue mb-3"),
            dcc.Dropdown(id='value-filter', options=[], placeholder="Filter values", className="dropdown-green mb-3"),
            dcc.Dropdown(id='y-axis-dropdown', options=[], placeholder="Select Y-axis", className="dropdown-orange mb-3"),
            dcc.Dropdown(id='chart-type', options=[
                {'label': 'Scatter', 'value': 'scatter'},
                {'label': 'Bar', 'value': 'bar'},
                {'label': 'Line', 'value': 'line'}
            ], placeholder="Select chart type", className="dropdown-purple mb-3"),
            dcc.Checklist(id='dark-mode-toggle', options=[{'label': 'Dark Mode', 'value': 'dark'}], value=['dark'], className="mb-3"),
            dcc.Checklist(id='outlier-toggle', options=[{'label': 'Show Outliers', 'value': 'outliers'}], value=[], className="mb-3"),
            dcc.Input(id='cluster-count', type='number', placeholder="Number of clusters", value=3, className="mb-3"),
            dcc.Dropdown(id='transform-type', options=[
                {'label': 'None', 'value': 'none'},
                {'label': 'Log', 'value': 'log'},
                {'label': 'Normalize', 'value': 'normalize'}
            ], placeholder="Transform type", className="mb-3"),
            html.Button('Clear Selection', id='clear-selection', className="btn btn-outline-red btn-block mb-3"),
            html.Button('Download Data', id='download-button', className="btn btn-outline-cyan btn-block mb-3"),
            html.Button('Download Charts', id='download-chart-btn', className="btn btn-outline-cyan btn-block mb-3"),
        ])
    ], className="bg-card-dark border-0 shadow-sm")
], width=3, className="sidebar collapse show", id="sidebar")

# Main content
main_content = dbc.Col([
    dbc.NavbarSimple(brand="📊 Advanced Data Analysis Dashboard", color="dark", dark=True, className="mb-4 navbar-dark"),
    dcc.Graph(id='main-chart', className="mb-4"),
    dcc.Graph(id='pie-chart', className="mb-4"),
    dcc.Graph(id='line-chart', className="mb-4"),
    dcc.Graph(id='main-chart-all', className="mb-4"),
    dcc.Graph(id='pie-chart-all', className="mb-4"),
    dcc.Graph(id='line-chart-all', className="mb-4"),
    html.Div(id='insights-panel', className="text-light mb-4"),
    dag.AgGrid(id='data-table', className="ag-theme-alpine-dark mb-4", rowData=[], columnDefs=[]),
    html.Div(id='upload-message', className="text-light mb-4"),
    html.Div(id='summary-stats', className="text-light mb-4"),
    dcc.Store(id='download-charts-store'),
    dcc.Download(id="download-charts"),
    dcc.Download(id="download-dataframe-csv"),
    dcc.Store(id='selected-category'),
    dcc.Interval(id='interval', interval=60000, n_intervals=0),
    # Modal for enlarged chart
    dbc.Modal([
        dbc.ModalHeader("Enlarged Chart", className="modal-dark"),
        dbc.ModalBody(dcc.Graph(id='enlarged-chart')),
        dbc.ModalFooter(
            html.Button("Close", id="close-modal", className="btn btn-outline-red")
        ),
    ], id="chart-modal", size="lg", className="modal-dark", is_open=False),
], width=9)

# Layout
app.layout = dbc.Container([
    dbc.Row([sidebar, main_content], className="min-vh-100")
], fluid=True, className="bg-gradient-dark")

# Main callback
@app.callback(
    [Output('column-filter', 'options'), Output('value-filter', 'options'), Output('y-axis-dropdown', 'options'),
     Output('main-chart', 'figure'), Output('pie-chart', 'figure'), Output('line-chart', 'figure'),
     Output('main-chart-all', 'figure'), Output('pie-chart-all', 'figure'), Output('line-chart-all', 'figure'),
     Output('insights-panel', 'children'), Output('data-table', 'rowData'), Output('data-table', 'columnDefs'),
     Output('upload-message', 'children'), Output('summary-stats', 'children'), Output('selected-category', 'data'),
     Output('download-charts-store', 'data')],
    [Input('upload-data', 'contents'), Input('column-filter', 'value'), Input('value-filter', 'value'),
     Input('y-axis-dropdown', 'value'), Input('chart-type', 'value'), Input('dark-mode-toggle', 'value'),
     Input('main-chart', 'clickData'), Input('clear-selection', 'n_clicks'), Input('outlier-toggle', 'value'),
     Input('cluster-count', 'value'), Input('transform-type', 'value'), Input('interval', 'n_intervals')],
    [State('upload-data', 'filename')]
)
def update_dashboard(contents, selected_column, selected_values, selected_y_axis, chart_type, dark_mode, click_data, clear_clicks, show_outliers, cluster_count, transform_type, n_intervals, filename):
    dark_mode = 'dark' in (dark_mode or [])
    show_outliers = 'outliers' in (show_outliers or [])

    # Default outputs
    column_options = []
    value_options = []
    y_axis_options = []
    main_fig = px.scatter(title="Please select X and Y axes")
    pie_fig = px.scatter(title="No pie chart generated")
    line_fig = px.scatter(title="No line chart generated")
    main_fig_all = px.scatter(title="All Data: Select X and Y axes")
    pie_fig_all = px.scatter(title="All Data: No pie chart generated")
    line_fig_all = px.scatter(title="All Data: No line chart generated")
    insights = []
    row_data = []
    column_defs = []
    upload_message = "No file uploaded yet."
    summary_stats = "No data loaded."
    selected_category = None
    charts_data = {}

    # Parse uploaded file
    if contents:
        df = parse_contents(contents, filename)
        if df is not None:
            upload_message = f"Uploaded: {filename} at {data_store.last_updated}"
            column_options = [{'label': col, 'value': col} for col in df.columns]
            y_axis_options = column_options

            # Filter data
            filtered_df = df.copy()
            if selected_column and selected_values:
                filtered_df = filtered_df[filtered_df[selected_column].isin(selected_values)]
            data_store.filtered_df = filtered_df

            # Update value filter options
            if selected_column:
                value_options = [{'label': str(val), 'value': val} for val in filtered_df[selected_column].unique()]

            # Generate charts for filtered data
            if selected_column and selected_y_axis:
                main_fig = generate_main_chart(filtered_df, selected_column, selected_y_axis, chart_type, dark_mode, selected_category, show_outliers, cluster_count, transform_type)
                pie_fig = generate_pie_chart(filtered_df, selected_column, selected_y_axis, dark_mode, selected_category, transform_type)
                line_fig = generate_line_chart(filtered_df, selected_column, selected_y_axis, dark_mode, selected_category, transform_type)

            # Generate charts for all data
            if selected_column and selected_y_axis:
                main_fig_all = generate_main_chart(df, selected_column, selected_y_axis, chart_type, dark_mode, None, show_outliers, cluster_count, transform_type)
                pie_fig_all = generate_pie_chart(df, selected_column, selected_y_axis, dark_mode, None, transform_type)
                line_fig_all = generate_line_chart(df, selected_column, selected_y_axis, dark_mode, None, transform_type)

            # Generate insights
            if selected_column and selected_y_axis:
                insights.append(html.P(f"Mean {selected_y_axis}: {filtered_df[selected_y_axis].mean():.2f}"))
                insights.append(html.P(f"Std Dev {selected_y_axis}: {filtered_df[selected_y_axis].std():.2f}"))

            # Update data table
            row_data = filtered_df.to_dict('records')
            column_defs = [{'field': col} for col in filtered_df.columns]

            # Summary stats
            summary_stats = [
                html.P(f"Rows: {len(filtered_df)}"),
                html.P(f"Columns: {len(filtered_df.columns)}")
            ]

            # Save charts for download
            charts = {'main': main_fig, 'pie': pie_fig, 'line': line_fig}
            for chart_name, fig in charts.items():
                img_bytes = fig.write_image("temp.png", engine="kaleido")
                charts_data[chart_name] = base64.b64encode(img_bytes).decode('utf-8')

    # Handle click data
    if click_data and 'points' in click_data:
        selected_category = click_data['points'][0].get('customdata', None)

    # Clear selection
    if clear_clicks:
        selected_category = None

    return (column_options, value_options, y_axis_options, main_fig, pie_fig, line_fig,
            main_fig_all, pie_fig_all, line_fig_all, insights, row_data, column_defs,
            upload_message, summary_stats, selected_category, charts_data)

# Modal callback
@app.callback(
    [Output('chart-modal', 'is_open'), Output('enlarged-chart', 'figure')],
    [Input('main-chart-all', 'clickData'), Input('pie-chart-all', 'clickData'), Input('line-chart-all', 'clickData'), Input('close-modal', 'n_clicks')],
    [State('main-chart-all', 'figure'), State('pie-chart-all', 'figure'), State('line-chart-all', 'figure'), State('chart-modal', 'is_open')]
)
def toggle_modal(main_click, pie_click, line_click, close_click, main_fig, pie_fig, line_fig, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open, go.Figure()

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'close-modal':
        return False, go.Figure()

    if trigger_id == 'main-chart-all' and main_click:
        return True, main_fig
    elif trigger_id == 'pie-chart-all' and pie_click:
        return True, pie_fig
    elif trigger_id == 'line-chart-all' and line_click:
        return True, line_fig
    return is_open, go.Figure()

# Download callbacks
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_data(n_clicks):
    if data_store.df is not None:
        return dcc.send_data_frame(data_store.df.to_csv, "dashboard_data.csv")
    return None

@app.callback(
    Output("download-charts", "data"),
    Input("download-chart-btn", "n_clicks"),
    State("download-charts-store", "data"),
    prevent_initial_call=True
)
def download_charts(n_clicks, charts_data):
    if n_clicks and charts_data:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for chart_name, chart_base64 in charts_data.items():
                chart_bytes = base64.b64decode(chart_base64)
                zip_file.writestr(f"{chart_name}_chart.png", chart_bytes)
        buffer.seek(0)
        return dcc.send_bytes(buffer.getvalue(), "charts.zip")
    return None

# Custom index string with PWA support
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="manifest" href="/static/manifest.json">
        <style>
            .bg-gradient-dark { background: linear-gradient(135deg, #1f2a44 0%, #2e3b55 100%); }
            .bg-card-dark { background-color: #2e3b55; }
            .navbar-dark { background-color: #1f2a44 !important; }
            .sidebar { transition: all 0.3s ease; }
            .sidebar.collapsed { margin-left: -25%; }
            .card { border-radius: 12px; transition: transform 0.2s ease, box-shadow 0.2s ease; border: 1px solid #ffffff10; }
            .card:hover { transform: translateY(-5px); box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2); }
            .btn-outline-cyan { border-color: #00d4ff; color: #00d4ff; transition: background-color 0.2s ease; }
            .btn-outline-cyan:hover { background-color: #00d4ff20; color: #00d4ff; }
            .btn-outline-red { border-color: #ff4d4f; color: #ff4d4f; transition: background-color 0.2s ease; }
            .btn-outline-red:hover { background-color: #ff4d4f20; color: #ff4d4f; }
            .tabs-custom .nav-link { color: #ffffff80; border-radius: 8px 8px 0 0; transition: background-color 0.2s ease; }
            .tabs-custom .nav-link.active { color: #ffffff; background-color: #2e3b55; border-bottom: 3px solid #00d4ff; }
            .tabs-custom .nav-link:hover { color: #ffffff; background-color: #ffffff10; }
            .dropdown-blue .Select-control { background-color: #4682b4 !important; border-color: #4682b4 !important; }
            .dropdown-green .Select-control { background-color: #2ecc71 !important; border-color: #2ecc71 !important; }
            .dropdown-orange .Select-control { background-color: #e67e22 !important; border-color: #e67e22 !important; }
            .dropdown-purple .Select-control { background-color: #9b59b6 !important; border-color: #9b59b6 !important; }
            .Select-menu-outer { background-color: #2e3b55 !important; border-color: #ffffff20 !important; }
            .Select-option { color: #ffffff !important; }
            .Select-option.is-focused { background-color: #00d4ff20 !important; }
            .text-light { color: #e6e6e6 !important; }
            .modal-dark .modal-content { background-color: #2e3b55; border: 1px solid #ffffff10; animation: fadeIn 0.3s ease; }
            .modal-dark .modal-header { border-bottom: 1px solid #ffffff20; }
            .modal-dark .modal-footer { border-top: 1px solid #ffffff20; }
            @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
            <script>
                if ('serviceWorker' in navigator) {
                    window.addEventListener('load', () => {
                        navigator.serviceWorker.register('/static/sw.js')
                            .then(reg => console.log('Service Worker registered'))
                            .catch(err => console.log('Service Worker registration failed: ', err));
                    });
                }
            </script>
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=port)