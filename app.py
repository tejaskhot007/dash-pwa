import base64
import io
import logging
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import datetime
import chardet
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

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
        {"name": "apple-mobile-web-app-capable", "content": "yes"},
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
        
        # Clean the data
        df = clean_data(df)
        if df is None:
            logger.error("Failed to clean data")
            return None
        
        # Sample the dataset to reduce load (200 rows instead of 500)
        if len(df) > 200:
            df = df.sample(n=200, random_state=42)
            logger.info(f"Dataset sampled to 200 rows: {df.shape}")
        
        logger.info(f"File parsed successfully: {df.shape}")
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
            html.Button("Clear Selection", id="clear-selection", className="btn btn-outline-red btn-block mb-3"),
            html.H5("Insights", className="card-title text-light"),
            html.Div(id='insights-panel', style={'maxHeight': '200px', 'overflowY': 'auto'})
        ])
    ], className="bg-card-dark border-0 shadow-sm")
], width=3, className="sidebar collapse show", id="sidebar")

# Main content with summary statistics, data table, and modal
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
            html.Div(id='data-table-debug'),  # Debug output for rows and columns
            dag.AgGrid(
                id='data-table',
                defaultColDef={
                    "resizable": True,
                    "sortable": True,
                    "filter": True,
                    "editable": False,
                    "wrapText": True,
                    "autoHeight": True,
                    "minWidth": 100,
                },
                dashGridOptions={
                    "pagination": True,
                    "paginationPageSize": 10,
                    "paginationPageSizeSelector": [10, 20, 50, 100],  # Ensure 10 is included
                    "animateRows": True,
                    "suppressRowClickSelection": True,
                    "rowBuffer": 0,
                    "enableCellTextSelection": True,
                    "domLayout": "normal",  # Ensure proper layout
                },
                className="ag-theme-alpine-dark",
                style={'height': '400px', 'width': '100%', 'overflow': 'auto', 'border': '1px solid #ccc'},
                loading_state={'is_loading': False, 'component_name': 'data-table'}  # Explicitly set loading_state
            )
        ])
    ], className="bg-card-dark border-0 shadow-sm"),
    dcc.Download(id="download-dataframe-csv"),
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Enlarged Chart"), className="bg-card-dark text-light"),
        dbc.ModalBody(dcc.Graph(id='enlarged-chart', style={'height': '70vh'}), className="bg-card-dark"),
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

# Main callback with chart highlighting and deselection
@app.callback(
    [Output('column-filter', 'options'),
     Output('value-filter', 'options'),
     Output('y-axis-dropdown', 'options'),
     Output('main-chart', 'figure'),
     Output('pie-chart', 'figure'),
     Output('line-chart', 'figure'),
     Output('insights-panel', 'children'),
     Output('data-table', 'rowData'),
     Output('data-table', 'columnDefs'),
     Output('upload-message', 'children'),
     Output('summary-stats', 'children'),
     Output('selected-category', 'data'),
     Output('data-table-debug', 'children')],  # Debug output
    [Input('upload-data', 'contents'),
     Input('column-filter', 'value'),
     Input('value-filter', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('chart-type', 'value'),
     Input('dark-mode-toggle', 'value'),
     Input('main-chart', 'clickData'),
     Input('clear-selection', 'n_clicks')],
    [State('upload-data', 'filename')]
)
def update_dashboard(contents, selected_column, selected_values, selected_y_axis, chart_type, dark_mode, click_data, clear_clicks, filename):
    try:
        logger.info("update_dashboard callback triggered")
        logger.info(f"Contents: {contents is not None}, Filename: {filename}")

        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        # Default outputs
        default_fig = px.scatter(title="Please upload a file")
        default_outputs = [[] for _ in range(3)] + [default_fig] * 3 + [[] for _ in range(3)] + ["Please upload a file", "No data available", None, "No data"]

        if not contents and data_store.df is None:
            logger.info("No contents and no stored data, returning default outputs")
            return default_outputs

        if triggered_id == 'upload-data':
            logger.info("Processing new upload")
            df = parse_contents(contents, filename)
            if df is None:
                logger.error("Failed to parse contents")
                default_fig = px.scatter(title="Error loading file")
                return [[] for _ in range(3)] + [default_fig] * 3 + [[] for _ in range(3)] + ["❌ Error loading file", "Error loading data", None, "Error"]
            data_store.df = df
        else:
            logger.info("Using stored DataFrame")
            df = data_store.df

        if df is None:
            logger.error("DataFrame is None")
            default_fig = px.scatter(title="No data available")
            return [[] for _ in range(3)] + [default_fig] * 3 + [[] for _ in range(3)] + ["No data available", "No data available", None, "No data"]

        logger.info(f"DataFrame shape after loading: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")

        logger.info("Selecting categorical and numerical columns")
        categorical_cols = df.select_dtypes(include=['object', 'datetime']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

        # Set default selections if none are provided
        selected_column = selected_column or (categorical_cols[0] if categorical_cols else None)
        selected_y_axis = selected_y_axis or (numerical_cols[0] if numerical_cols else None)
        chart_type = chart_type or 'bar'

        logger.info("Generating value options for dropdown")
        value_options = ([{'label': str(val), 'value': val} 
                         for val in df[selected_column].dropna().unique()] 
                         if selected_column in df.columns else [])
        
        logger.info("Filtering DataFrame")
        df_filtered = df[df[selected_column].isin(selected_values)] if selected_column and selected_values else df.copy()
        data_store.filtered_df = df_filtered  # Cache filtered data
        logger.info(f"Filtered DataFrame shape: {df_filtered.shape}")
        logger.info(f"Filtered DataFrame columns: {df_filtered.columns.tolist()}")

        # Handle selection and deselection
        selected_category = None
        if triggered_id == 'main-chart' and click_data and chart_type in ['bar', 'scatter', 'line', 'regression']:
            logger.info("Handling main chart click data")
            selected_category = click_data['points'][0].get('x') or click_data['points'][0].get('label')
        elif triggered_id == 'clear-selection' and clear_clicks:
            logger.info("Clearing selection")
            selected_category = None

        # Generate charts for main tabs
        logger.info("Generating main chart")
        main_fig = generate_main_chart(df_filtered, selected_column, selected_y_axis, chart_type, dark_mode, selected_category)
        logger.info("Generating pie chart")
        pie_fig = generate_pie_chart(df_filtered, selected_column, selected_y_axis, dark_mode, selected_category)
        logger.info("Generating line chart")
        line_fig = generate_line_chart(df_filtered, selected_column, selected_y_axis, dark_mode, selected_category)

        # Insights
        logger.info("Generating insights")
        insights = [html.P(f"🔹 {col}: {df[col].nunique()} unique | Type: {df[col].dtype}", className="text-light") 
                    for col in df.columns]

        # Summary statistics
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

        # Prepare data for the table
        logger.info("Converting DataFrame to string for AG Grid")
        df_filtered = df_filtered.astype(str)
        logger.info("Preparing row data")
        row_data = df_filtered.to_dict('records')
        logger.info(f"Row data prepared: {len(row_data)} rows")
        logger.info(f"Sample row data: {row_data[:2] if row_data else 'Empty'}")

        # Generate column definitions for AG Grid
        logger.info("Generating column definitions")
        if df_filtered.empty:
            column_defs = [
                {
                    'field': 'message',
                    'headerName': 'Message',
                    'filter': 'agTextColumnFilter',
                    'sortable': True,
                    'resizable': True,
                    'editable': False
                }
            ]
            row_data = [{'message': 'No data available after filtering'}]
            logger.warning("DataFrame is empty, displaying placeholder message in table")
        else:
            column_defs = [
                {
                    'field': col,
                    'headerName': col.replace('_', ' ').title(),
                    'filter': 'agTextColumnFilter',
                    'sortable': True,
                    'resizable': True,
                    'editable': False
                } for col in df_filtered.columns
            ]
        logger.info(f"Column definitions generated: {column_defs}")

        # Debug output
        debug_info = f"Rows: {len(row_data)}, Columns: {len(column_defs)}"

        logger.info("Returning callback outputs")
        return (
            [{'label': col, 'value': col} for col in categorical_cols],
            value_options,
            [{'label': col, 'value': col} for col in numerical_cols],
            main_fig,
            pie_fig,
            line_fig,
            insights,
            row_data,
            column_defs,
            f"✅ Uploaded {filename} ({len(df)} rows)" if contents else f"Data loaded ({len(df)} rows)",
            summary_stats,
            selected_category,
            debug_info
        )
    except Exception as e:
        logger.error(f"Dashboard update error: {str(e)}", exc_info=True)
        default_fig = px.scatter(title=f"Dashboard error: {str(e)}")
        return [[] for _ in range(3)] + [default_fig] * 3 + [[] for _ in range(3)] + [f"❌ Error: {str(e)}", "Error loading data", None, "Error"]

# Separate callback for "All Charts" tab to reduce load
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
            return [px.scatter()] * 3  # Return empty figures if not on "All Charts" tab

        if data_store.df is None:
            return [px.scatter(title="No data available")] * 3

        df = data_store.df
        df_filtered = df[df[selected_column].isin(selected_values)] if selected_column and selected_values else df.copy()

        main_fig_all = generate_main_chart(df, selected_column, selected_y_axis, chart_type, dark_mode, None)
        pie_fig_all = generate_pie_chart(df, selected_column, selected_y_axis, dark_mode, None)
        line_fig_all = generate_line_chart(df, selected_column, selected_y_axis, dark_mode, None)

        return main_fig_all, pie_fig_all, line_fig_all
    except Exception as e:
        logger.error(f"All charts update error: {str(e)}")
        return [px.scatter(title=f"Error: {str(e)}")] * 3

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

# Download callback
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True
)
def download_data(n_clicks):
    if data_store.df is not None:
        return dcc.send_data_frame(data_store.df.to_csv, "dashboard_data.csv")
    return None

# Custom CSS for optimized color combination and animations
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
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
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)