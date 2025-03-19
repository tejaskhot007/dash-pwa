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
        logger.info(f"Original columns: {df.columns.tolist()}")
        df.columns = (df.columns.str.strip()
                     .str.replace(" ", "_", regex=False)
                     .str.replace(r"[^\w\s]", "", regex=True)
                     .str.lower())
        logger.info(f"Cleaned columns: {df.columns.tolist()}")
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
        logger.info(f"Parsing file: {filename}")
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        encoding = chardet.detect(decoded)['encoding'] or 'utf-8'
        logger.info(f"Detected encoding: {encoding}")
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode(encoding)), low_memory=False, engine='c')
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')
        else:
            raise ValueError("Unsupported file format")
        df = clean_data(df)
        if df is not None:
            # Sample data to 500 rows
            if len(df) > 500:
                df = df.sample(n=500, random_state=42)
                logger.info(f"Dataset sampled to 500 rows: {df.shape}")
            data_store.df = df
            data_store.last_updated = datetime.now()
            data_store.filtered_df = None  # Reset filtered data
            logger.info(f"File parsed successfully: {df.shape}")
        else:
            logger.error("Data cleaning returned None")
        return df
    except Exception as e:
        logger.error(f"File parsing error: {e}")
        return None

# Chart generation functions
def generate_main_chart(df, selected_column, selected_y_axis, chart_type, dark_mode, selected_category):
    try:
        if not selected_column or not selected_y_axis:
            return px.scatter(title="Please select X and Y axes")
        
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

# Sidebar content with clear button
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
            html.Div(id='upload-message', className="mb-3 text-center text-light"),
            dcc.Dropdown(id='column-filter', placeholder="Select a column", className="mb-3 dropdown-blue"),
            dcc.Dropdown(id='value-filter', placeholder="Select values", multi=True, className="mb-3 dropdown-green"),
            dcc.Dropdown(id='y-axis-dropdown', placeholder="Select Y-axis", className="mb-3 dropdown-orange"),
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
            dbc.Switch(id='dark-mode-toggle', label='Dark Mode', value=True, className="mb-3 text-light"),
            html.Button("Download Data 📥", id="download-button", className="btn btn-outline-cyan btn-block mb-3"),
            html.Button("Download Chart Data 📊", id="download-chart-btn", className="btn btn-outline-cyan btn-block mb-3"),
            html.Button("Clear Selection", id="clear-selection", className="btn btn-outline-red btn-block mb-3"),
            html.H5("Insights", className="card-title text-light"),
            html.Div(id='insights-panel', style={'maxHeight': '200px', 'overflowY': 'auto'})
        ])
    ], className="bg-card-dark border-0 shadow-sm")
], width=3, className="sidebar collapse show", id="sidebar")

# Main content with summary statistics and modal
main_content = dbc.Col([
    dbc.NavbarSimple(brand="📊 Advanced Data Analysis Dashboard", color="dark", dark=True, className="mb-4 navbar-dark"),
    dbc.Card([
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab(label="Main Chart", tab_id="main-tab", children=[
                    dcc.Graph(id='main-chart', style={'height': '400px'})
                ]),
                dbc.Tab(label="Pie Chart", tab_id="pie-tab", children=[
                    dcc.Graph(id='pie-chart', style={'height': '400px'})
                ]),
                dbc.Tab(label="Line Chart", tab_id="line-tab", children=[
                    dcc.Graph(id='line-chart', style={'height': '400px'})
                ]),
                dbc.Tab(label="All Charts", tab_id="all-tab", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='main-chart-all', style={'height': '300px'}), width=4),
                        dbc.Col(dcc.Graph(id='pie-chart-all', style={'height': '300px'}), width=4),
                        dbc.Col(dcc.Graph(id='line-chart-all', style={'height': '300px'}), width=4)
                    ])
                ])
            ], id="chart-tabs", active_tab="main-tab", className="tabs-custom")
        ])
    ], className="bg-card-dark border-0 shadow-sm mb-4"),
    dbc.Card([
        dbc.CardBody([
            html.H5("Summary Statistics", className="card-title text-light"),
            html.Div(id='summary-stats')
        ])
    ], className="bg-card-dark border-0 shadow-sm mb-4"),
    dbc.Card([
        dbc.CardBody([
            html.H5("Data Table", className="card-title text-light"),
            dag.AgGrid(
                id='data-table',
                defaultColDef={"resizable": True, "sortable": True, "filter": True, "editable": False},
                dashGridOptions={"pagination": True, "paginationPageSize": 10},
                className="ag-theme-alpine-dark",
                style={'height': '400px', 'width': '100%'}
            )
        ])
    ], className="bg-card-dark border-0 shadow-sm"),
    dcc.Download(id="download-dataframe-csv"),
    dcc.Download(id="download-charts"),
    dcc.Store(id='selected-category'),
    dcc.Store(id='upload-timestamp', data=0),  # Store to track upload events
    dcc.Store(id='main-chart-store'),
    dcc.Store(id='pie-chart-store'),
    dcc.Store(id='line-chart-store'),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Enlarged Chart"), className="bg-card-dark text-light"),
        dbc.ModalBody(dcc.Graph(id='enlarged-chart', style={'height': '70vh'}), className="bg-card-dark"),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="btn btn-outline-cyan")
        )
    ], id="chart-modal", size="xl", is_open=False, backdrop=True, className="modal-dark")
], width=9)

# Layout
app.layout = dbc.Container([
    dbc.Row([
        sidebar,
        main_content
    ], className="min-vh-100")
], fluid=True, className="bg-gradient-dark")

# Callback to update timestamp on upload
@app.callback(
    Output('upload-timestamp', 'data'),
    Input('upload-data', 'contents'),
    State('upload-timestamp', 'data')
)
def update_timestamp(contents, current_timestamp):
    if contents:
        return current_timestamp + 1
    return current_timestamp

# Main callback with chart highlighting and deselection
@app.callback(
    [Output('column-filter', 'options'),
     Output('value-filter', 'options'),
     Output('y-axis-dropdown', 'options'),
     Output('main-chart', 'figure'),
     Output('pie-chart', 'figure'),
     Output('line-chart', 'figure'),
     Output('main-chart-all', 'figure'),
     Output('pie-chart-all', 'figure'),
     Output('line-chart-all', 'figure'),
     Output('insights-panel', 'children'),
     Output('data-table', 'rowData'),
     Output('data-table', 'columnDefs'),
     Output('upload-message', 'children'),
     Output('summary-stats', 'children'),
     Output('selected-category', 'data'),
     Output('main-chart-store', 'data'),
     Output('pie-chart-store', 'data'),
     Output('line-chart-store', 'data'),
     Output('column-filter', 'value'),  # Reset dropdown
     Output('value-filter', 'value'),   # Reset dropdown
     Output('y-axis-dropdown', 'value')],  # Reset dropdown
    [Input('upload-data', 'contents'),
     Input('upload-timestamp', 'data'),  # Trigger on upload event
     Input('column-filter', 'value'),
     Input('value-filter', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('chart-type', 'value'),
     Input('dark-mode-toggle', 'value'),
     Input('main-chart', 'clickData'),
     Input('clear-selection', 'n_clicks')],
    [State('upload-data', 'filename')]
)
def update_dashboard(contents, upload_timestamp, selected_column, selected_values, selected_y_axis, chart_type, dark_mode, click_data, clear_clicks, filename):
    try:
        logger.info("update_dashboard callback triggered")
        logger.info(f"Contents: {contents is not None}, Filename: {filename}, Upload Timestamp: {upload_timestamp}")
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        # Default outputs
        default_fig = px.scatter(title="Please upload a file")
        default_outputs = [[] for _ in range(3)] + [default_fig] * 6 + [[] for _ in range(3)] + ["Please upload a file", "No data available", None]

        # Reset dropdown values on new upload
        reset_column = None
        reset_values = []
        reset_y_axis = None

        if not contents and data_store.df is None:
            return default_outputs + [None, None, None] + [reset_column, reset_values, reset_y_axis]

        if triggered_id in ['upload-data', 'upload-timestamp']:
            df = parse_contents(contents, filename)
            if df is None:
                default_fig = px.scatter(title="Error loading file")
                return [[] for _ in range(3)] + [default_fig] * 6 + [[] for _ in range(3)] + ["❌ Error loading file", "Error loading data", None] + [None, None, None] + [reset_column, reset_values, reset_y_axis]
            data_store.df = df
            # Reset selections on new upload
            selected_column = None
            selected_values = []
            selected_y_axis = None
        else:
            df = data_store.df

        if df is None:
            default_fig = px.scatter(title="No data available")
            return [[] for _ in range(3)] + [default_fig] * 6 + [[] for _ in range(3)] + ["No data available", "No data available", None] + [None, None, None] + [reset_column, reset_values, reset_y_axis]

        categorical_cols = df.select_dtypes(include=['object', 'datetime']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

        # Set default selections if none are provided
        selected_column = selected_column or (categorical_cols[0] if categorical_cols else None)
        selected_y_axis = selected_y_axis or (numerical_cols[0] if numerical_cols else None)
        chart_type = chart_type or 'bar'

        value_options = ([{'label': str(val), 'value': val} 
                         for val in df[selected_column].dropna().unique()] 
                         if selected_column in df.columns else [])
        
        df_filtered = df[df[selected_column].isin(selected_values)] if selected_column and selected_values else df.copy()
        data_store.filtered_df = df_filtered  # Cache filtered data

        # Handle selection and deselection
        selected_category = None
        if triggered_id == 'main-chart' and click_data and chart_type in ['bar', 'scatter', 'line', 'regression']:
            selected_category = click_data['points'][0].get('x') or click_data['points'][0].get('label')
        elif triggered_id == 'clear-selection' and clear_clicks:
            selected_category = None

        # Generate charts
        main_fig = generate_main_chart(df_filtered, selected_column, selected_y_axis, chart_type, dark_mode, selected_category)
        pie_fig = generate_pie_chart(df_filtered, selected_column, selected_y_axis, dark_mode, selected_category)
        line_fig = generate_line_chart(df_filtered, selected_column, selected_y_axis, dark_mode, selected_category)

        # Generate charts for all data
        main_fig_all = generate_main_chart(df, selected_column, selected_y_axis, chart_type, dark_mode, None)
        pie_fig_all = generate_pie_chart(df, selected_column, selected_y_axis, dark_mode, None)
        line_fig_all = generate_line_chart(df, selected_column, selected_y_axis, dark_mode, None)

        # Insights
        insights = [html.P(f"🔹 {col}: {df[col].nunique()} unique | Type: {df[col].dtype}", className="text-light") 
                    for col in df.columns]

        # Summary statistics
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

        # Prepare data for the table
        row_data = df_filtered.to_dict('records')
        logger.info(f"Row data prepared: {len(row_data)} rows")

        # Generate column definitions for AG Grid
        if df_filtered.empty:
            column_defs = []
            logger.warning("DataFrame is empty, no columns to display in table")
        else:
            column_defs = [
                {
                    'field': col,
                    'headerName': col.replace('_', ' ').title(),  # Human-readable header
                    'filter': 'agTextColumnFilter' if df_filtered[col].dtype == 'object' else 'agNumberColumnFilter',
                    'sortable': True,
                    'resizable': True,
                    'editable': False
                } for col in df_filtered.columns
            ]
            logger.info(f"Column definitions generated: {column_defs}")

        return (
            [{'label': col, 'value': col} for col in categorical_cols],
            value_options,
            [{'label': col, 'value': col} for col in numerical_cols],
            main_fig,
            pie_fig,
            line_fig,
            main_fig_all,
            pie_fig_all,
            line_fig_all,
            insights,
            row_data,
            column_defs,
            f"✅ Uploaded {filename} ({len(df)} rows)" if contents else f"Data loaded ({len(df)} rows)",
            summary_stats,
            selected_category,
            main_fig,
            pie_fig,
            line_fig,
            selected_column,
            selected_values,
            selected_y_axis
        )
    except Exception as e:
        logger.error(f"Dashboard update error: {e}")
        default_fig = px.scatter(title=f"Dashboard error: {str(e)}")
        return [[] for _ in range(3)] + [default_fig] * 6 + [[] for _ in range(3)] + [f"❌ Error: {str(e)}", "Error loading data", None] + [None, None, None] + [None, [], None]

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
    State("main-chart-store", "data"),
    State("pie-chart-store", "data"),
    State("line-chart-store", "data"),
    prevent_initial_call=True
)
def download_charts(n_clicks, main_fig, pie_fig, line_fig):
    if not n_clicks:
        return None
    
    # Download chart data as JSON
    charts_data = {
        'main_chart': main_fig,
        'pie_chart': pie_fig,
        'line_chart': line_fig
    }
    return dict(content=str(charts_data), filename="chart_data.json")

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
            /* Background Gradient */
            .bg-gradient-dark {
                background: linear-gradient(135deg, #1f2a44 0%, #2e3b55 100%);
            }
            /* Card Background */
            .bg-card-dark {
                background-color: #2e3b55;
            }
            /* Navbar */
            .navbar-dark {
                background-color: #1f2a44 !important;
            }
            /* Sidebar */
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
            /* Buttons */
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
            /* Tabs */
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
            /* Colorful Dropdowns */
            .dropdown-blue .Select-control {
                background-color: #4682b4 !important;
                border-color: #4682b4 !important;
                transition: opacity 0.2s ease;
            }
            .dropdown-green .Select-control {
                background-color: #2ecc71 !important;
                border-color: #2ecc71 !important;
                transition: opacity 0.2s ease;
            }
            .dropdown-orange .Select-control {
                background-color: #e67e22 !important;
                border-color: #e67e22 !important;
                transition: opacity 0.2s ease;
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
            /* Text */
            .text-light {
                color: #e6e6e6 !important;
            }
            /* Modal */
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