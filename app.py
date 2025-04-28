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
import zipfile
import functools
import warnings
import re
try:
    import pdfkit
except ImportError:
    pdfkit = None
import plotly.io as pio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True, title="📊 Extraordinary Data Analysis Dashboard")
server = app.server

# Custom HTML template for the app
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

# Data storage class
class DataStore:
    def __init__(self):
        self.df = None
        self.last_updated = None
        self.filtered_df = None
        self.raw_df = None
        self.original_row_count = None

data_store = DataStore()

# Cache decorator for performance optimization
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

# Data cleaning and validation function
def clean_and_validate_data(df):
    try:
        if data_store.raw_df is None:
            data_store.raw_df = df.copy()
            data_store.original_row_count = len(df)
        simplified_messages = []
        
        # Identify potential date columns based on recognizable date patterns
        potential_date_cols = []
        for col in df.columns:
            sample = df[col].dropna().head(5).astype(str)
            if any('-' in s or '/' in s for s in sample) and not any(s.isdigit() and len(s) > 10 for s in sample):
                potential_date_cols.append(col)
        
        # Try multiple date formats explicitly
        date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d']
        for col in potential_date_cols:
            parsed = False
            for fmt in date_formats:
                try:
                    df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                    if df[col].notna().any():
                        logger.info(f"Successfully parsed {col} as datetime with format {fmt}")
                        parsed = True
                        break
                except Exception as e:
                    continue
            if not parsed:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')  # Fallback to dateutil
                    if df[col].notna().any():
                        logger.info(f"Parsed {col} as datetime using dateutil fallback")
                    else:
                        logger.warning(f"Could not parse {col} as datetime")
                except Exception as e:
                    logger.warning(f"Could not parse {col} as datetime: {e}")

        # Extract numeric values from object columns
        for col in df.columns:
            if col not in potential_date_cols and df[col].dtype == 'object':
                try:
                    # Extract numbers (integers or decimals, including negative)
                    df[col] = df[col].astype(str).apply(
                        lambda x: re.findall(r'-?\d*\.?\d+', x)[0] if re.findall(r'-?\d*\.?\d+', x) else np.nan
                    )
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].notna().sum() > len(df[col]) * 0.1:  # Require at least 10% non-NA
                        logger.info(f"Converted {col} to numeric by extracting numbers")
                    else:
                        df[col] = data_store.raw_df[col]  # Revert if mostly NA
                except Exception as e:
                    logger.warning(f"Could not extract numeric values from {col}: {e}")
                    df[col] = data_store.raw_df[col]

        # Clean object columns and preserve numeric columns
        numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        for col in df.select_dtypes(include=['object']).columns:
            if col not in potential_date_cols and col not in numerical_cols:
                df[col] = df[col].str.lower().str.strip()
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop columns that are entirely NA
        df = df.dropna(how='all', axis=1)
        
        # Handle missing values
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove duplicates
        df = df.drop_duplicates(keep='first')
        
        return df, simplified_messages
    except Exception as e:
        logger.error(f"Data cleaning error: {e}")
        return None, [f"Error during cleaning: {str(e)}"]

# File parsing function
def parse_contents(contents, filename, max_rows=None):
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
        df, cleaning_messages = clean_and_validate_data(df)
        if df is None:
            return None, cleaning_messages
        simplified_messages = cleaning_messages
        return df, simplified_messages
    except Exception as e:
        logger.error(f"File parsing error: {e}")
        return None, [f"Error parsing file: {str(e)}"]

# Main chart generation function
@cache
def generate_main_chart(df, x_axis, y_axes, chart_type, dark_mode, selected_category):
    try:
        logger.info(f"Generating main chart with x_axis: {x_axis}, y_axes: {y_axes}, chart_type: {chart_type}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Unique values in {x_axis}: {df[x_axis].unique() if x_axis in df.columns else 'N/A'}")
        
        if not x_axis or not y_axes or x_axis not in df.columns or not all(y in df.columns for y in y_axes):
            logger.error("Invalid X or Y axis selected")
            return px.scatter(title="Please select valid X and Y axes")

        if chart_type == 'bar' and not all(pd.api.types.is_numeric_dtype(df[y]) for y in y_axes):
            logger.error(f"One or more Y-axes {y_axes} are not numerical for bar chart")
            return px.scatter(title=f"Y-axes must be numerical for bar chart")

        if df.empty:
            logger.error("Data is empty")
            return px.scatter(title="Data is empty")

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
            if (pd.api.types.is_numeric_dtype(df[x_axis]) and pd.api.types.is_numeric_dtype(df[y_axis])):
                from sklearn.linear_model import LinearRegression
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

# Pie chart generation function
@cache
def generate_pie_chart(df, names_col, dark_mode, selected_category):
    try:
        if not names_col or names_col not in df.columns:
            return px.scatter(title="Please select a column for the Pie Chart")
        
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numerical_cols:
            return px.scatter(title="No numerical columns available for Pie Chart values")
        values_col = numerical_cols[0]

        if df.empty:
            logger.error("Data is empty")
            return px.scatter(title="Data is empty")

        fig = px.pie(df, names=names_col, values=values_col, template='plotly_dark' if dark_mode else 'plotly', title=f"Pie: {names_col} Distribution")
        if selected_category:
            explode = [0.1 if x == selected_category else 0 for x in df[names_col].unique()]
            fig.update_traces(pull=explode, marker=dict(colors=['#00d4ff' if x == selected_category else '#636efa' for x in df[names_col].unique()]))
        return fig
    except Exception as e:
        logger.error(f"Pie chart error: {e}")
        return px.scatter(title=f"Error generating pie chart: {str(e)}")

# Line chart generation function
@cache
def generate_line_chart(df, x_axis, y_axis, dark_mode, selected_category):
    try:
        if not x_axis or not y_axis or x_axis not in df.columns or y_axis not in df.columns:
            return px.scatter(title="Please select X and Y axes for the Line Chart")
        if df.empty:
            logger.error("Data is empty")
            return px.scatter(title="Data is empty")
        fig = px.line(df, x=x_axis, y=y_axis, template='plotly_dark' if dark_mode else 'plotly', title=f"Line: {x_axis} vs {y_axis}")
        if selected_category:
            fig.add_scatter(x=[selected_category], y=[df[df[x_axis] == selected_category][y_axis].iloc[0]], mode='markers', marker=dict(size=15, color='#00d4ff'), showlegend=False)
        return fig
    except Exception as e:
        logger.error(f"Line chart error: {e}")
        return px.scatter(title=f"Error generating line chart: {str(e)}")

# Correlation heatmap generation function
@cache
def generate_correlation_heatmap(df, dark_mode):
    try:
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numerical_cols) < 2:
            return px.scatter(title="Not enough numerical columns for correlation")
        if df.empty:
            logger.error("Data is empty")
            return px.scatter(title="Data is empty")
        corr_matrix = df[numerical_cols].corr()
        fig = ff.create_annotated_heatmap(z=corr_matrix.values, x=numerical_cols, y=numerical_cols, colorscale='Plasma' if dark_mode else 'Viridis', showscale=True)
        fig.update_layout(title="Correlation Heatmap", template='plotly_dark' if dark_mode else 'plotly')
        return fig
    except Exception as e:
        logger.error(f"Correlation heatmap error: {e}")
        return px.scatter(title=f"Error generating correlation heatmap: {str(e)}")

# Smart insights generation function
@cache
def generate_smart_insights(df, y_axis=None):
    try:
        insights = []
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if y_axis and y_axis in df.columns and pd.api.types.is_numeric_dtype(df[y_axis]):
            num_col = y_axis
        else:
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
            num_col = next((col for col in numerical_cols if any(k in col.lower() for k in ['revenue', 'sales', 'amount'])), None)
            if not num_col and numerical_cols:
                num_col = numerical_cols[0]
        
        if not num_col:
            return ["No suitable numerical column found for insights"]

        for cat_col in categorical_cols:
            if cat_col in df.columns and num_col in df.columns:
                top_performers = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(3)
                insights.append(html.P(f"🔹 Top 3 {cat_col} by {num_col}:", className="text-light"))
                insights.append(html.Ul([html.Li(f"{idx}: {val:.2f}") for idx, val in top_performers.items()], className="text-light"))

        return insights if insights else ["No smart insights available"]
    except Exception as e:
        logger.error(f"Smart insights error: {e}")
        return [f"Error generating smart insights: {str(e)}"]

# KPI cards generation function
def generate_kpi_cards(df):
    try:
        kpi_cards = []
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        revenue_col = next((col for col in numerical_cols if any(k in col.lower() for k in ['revenue', 'amount', 'sales'])), None)
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

# Executive summary generation function
def generate_executive_summary(df, kpi_cards, smart_insights):
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

# Sidebar layout
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
            dcc.Dropdown(id='column-filter', placeholder="Select a column to filter", className="mb-3 dropdown-purple"),
            dcc.Dropdown(id='value-filter', placeholder="Select values to filter", value=[], multi=True, className="mb-3 dropdown-purple"),
            dbc.Switch(id='dark-mode-toggle', label='Dark Mode', value=True, className="mb-3 text-light"),
            html.Button("Download Data 📥", id="download-button", className="btn btn-outline-cyan btn-block mb-3"),
            html.Button("Clear Selection", id="clear-selection", className="btn btn-outline-red btn-block mb-3"),
            html.H5("What-If Analysis", className="card-title text-light mt-3"),
            html.Label("Adjust Numerical Values (% Change)", className="text-light"),
            dcc.Dropdown(id='what-if-column', placeholder="Select a numerical column", className="mb-3 dropdown-purple"),
            dcc.Slider(id='what-if-adjust', min=-50, max=50, step=5, value=0, marks={i: f"{i}%" for i in range(-50, 51, 25)}),
            html.H5("Smart Insights", className="card-title text-light mt-3"),
            html.Div(id='smart-insights', style={'maxHeight': '200px', 'overflowY': 'auto'}),
            html.H5("Summary Statistics", className="card-title text-light mt-3"),
            html.Div(id='summary-stats'),
            html.H5("Export Report", className="card-title text-light mt-3"),
            html.Button("Generate Executive Summary 📄", id="export-report", className="btn btn-outline-cyan btn-block mb-3"),
            dcc.Download(id="download-report"),
        ])
    ], className="bg-card-dark border-0 shadow-sm")
], width=3, className="sidebar collapse show", id="sidebar")

# Main content layout
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
], width=9)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        sidebar,
        main_content
    ])
], fluid=True)

# Callback to update dropdowns after file upload
@app.callback(
    [Output('analysis-x-axis', 'options'),
     Output('analysis-y-axis', 'options'),
     Output('column-filter', 'options'),
     Output('what-if-column', 'options'),
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
        return [], [], [], [], [], [], [], "Please upload a file", "", None, [], [], None, []

    df, messages = parse_contents(contents, filename)
    if df is None:
        return [], [], [], [], [], [], [], messages, "", None, [], [], None, []

    data_store.df = df
    data_store.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    columns = [{'label': col, 'value': col} for col in df.columns]
    numerical_cols = [{'label': col, 'value': col} for col in df.select_dtypes(include=['number']).columns]

    table_columns = [{"name": col, "id": col} for col in df.columns]
    table_data = df.to_dict('records')

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        default_x_axis = categorical_cols[0]
    else:
        default_x_axis = df.columns[0]

    default_y_axis = numerical_cols[0]['value'] if numerical_cols else None

    upload_message = f"File {filename} contains {len(df)} rows" if df is not None else "Please upload a file"
    return (columns, numerical_cols, columns, numerical_cols,
            columns, columns, numerical_cols, upload_message, messages,
            df.to_dict('records'), table_data, table_columns, default_x_axis, [default_y_axis] if default_y_axis else [])

# Callback to update value filter options
@app.callback(
    Output('value-filter', 'options'),
    [Input('column-filter', 'value')],
    [State('filtered-data', 'data')]
)
def update_value_filter(column, data):
    if not column or not data:
        return []
    try:
        df = pd.DataFrame(data)
        if column not in df.columns:
            return []
        unique_values = df[column].dropna().unique()
        return [{'label': str(val), 'value': str(val)} for val in unique_values]
    except Exception as e:
        logger.error(f"Error updating value filter: {e}")
        return []

# Callback to update X-axis filter values
@app.callback(
    Output('x-axis-filter-values', 'options'),
    [Input('analysis-x-axis', 'value')],
    [State('filtered-data', 'data')]
)
def update_x_axis_filter_values(x_axis, data):
    if not x_axis or not data:
        return []
    try:
        df = pd.DataFrame(data)
        if x_axis not in df.columns:
            return []
        unique_values = df[x_axis].dropna().unique()
        return [{'label': str(val), 'value': str(val)} for val in unique_values]
    except Exception as e:
        logger.error(f"Error updating X-axis filter values: {e}")
        return []

# Callback to update KPI cards, smart insights, and filtered data
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
     Input('x-axis-filter-values', 'value'),
     Input('analysis-y-axis', 'value')],
    prevent_initial_call=True
)
def update_kpi_and_insights(data, filter_values, filter_column, analysis_x_axis, what_if_column, what_if_adjust, x_axis_filter_values, y_axes):
    if not data:
        return [], ["Please upload data to see insights"], None, []
    
    try:
        df = pd.DataFrame(data)
        logger.info(f"Initial DataFrame shape: {df.shape}")
        logger.info(f"Initial unique values in {analysis_x_axis}: {df[analysis_x_axis].unique() if analysis_x_axis in df.columns else 'N/A'}")
        
        # Apply column filter (if any)
        if filter_column and filter_values:
            filter_values = [str(val).lower().strip() for val in filter_values]
            df[filter_column] = df[filter_column].astype(str).str.lower().str.strip()
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
        y_axis = y_axes[0] if y_axes else None
        smart_insights = generate_smart_insights(df, y_axis)
        table_data = df.to_dict('records')
        logger.info(f"Final filtered DataFrame shape: {df.shape}")
        return kpi_cards, smart_insights, df.to_dict('records'), table_data
    except Exception as e:
        logger.error(f"Error in update_kpi_and_insights: {str(e)}")
        return [], [f"Error processing data: {str(e)}"], None, []

# Callback to update charts
@app.callback(
    [Output('main-chart', 'figure'),
     Output('pie-chart', 'figure'),
     Output('line-chart', 'figure'),
     Output('correlation-heatmap', 'figure')],
    [Input('tabs', 'active_tab'),
     Input('filtered-data', 'data'),
     Input('analysis-x-axis', 'value'),
     Input('analysis-y-axis', 'value'),
     Input('chart-type', 'value'),
     Input('dark-mode-toggle', 'value'),
     Input('selected-category', 'data'),
     Input('pie-names-column', 'value'),
     Input('line-x-axis', 'value'),
     Input('line-y-axis', 'value')]
)
def update_charts(active_tab, data, x_axis, y_axes, chart_type, dark_mode, selected_category, pie_names_col, line_x_axis, line_y_axis):
    empty_fig = px.scatter(title="Please upload data")
    if not data:
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    try:
        df = pd.DataFrame(data)
        if df.empty or df is None:
            empty_fig = px.scatter(title="No data available")
            return empty_fig, empty_fig, empty_fig, empty_fig
        
        main_fig = px.scatter(title="Select X and Y axes")
        pie_fig = px.scatter(title="Select a column for the Pie Chart")
        line_fig = px.scatter(title="Select X and Y axes for the Line Chart")
        heatmap_fig = px.scatter(title="Select numerical columns")
        
        if active_tab == "main-tab" and x_axis and y_axes:
            main_fig = generate_main_chart(df, x_axis, y_axes if y_axes else [], chart_type, dark_mode, selected_category)
        elif active_tab == "pie-tab" and pie_names_col:
            pie_fig = generate_pie_chart(df, pie_names_col, dark_mode, selected_category)
        elif active_tab == "line-tab" and line_x_axis and line_y_axis:
            line_fig = generate_line_chart(df, line_x_axis, line_y_axis, dark_mode, selected_category)
        elif active_tab == "correlation-tab":
            heatmap_fig = generate_correlation_heatmap(df, dark_mode)
        
        return main_fig, pie_fig, line_fig, heatmap_fig
    except Exception as e:
        logger.error(f"Error in update_charts: {str(e)}")
        error_fig = px.scatter(title=f"Chart update error: {str(e)}")
        return error_fig, error_fig, error_fig, error_fig

# Callback to update selected category
@app.callback(
    Output('selected-category', 'data'),
    [Input('main-chart', 'clickData')],
    [State('analysis-x-axis', 'value')]
)
def update_selected_category(click_data, x_axis):
    if not click_data or not x_axis:
        return None
    return click_data['points'][0]['x']

# Callback to download data
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

# Callback to download analysis charts
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
                    logger.error(f"Error saving {filename}: {e}")
                    continue

    zip_buffer.seek(0)
    return dict(
        content=base64.b64encode(zip_buffer.getvalue()).decode(),
        filename="analysis_charts.zip",
        type="application/zip",
        base64=True
    )

# Callback to clear selections
@app.callback(
    [Output('analysis-x-axis', 'value'),
     Output('analysis-y-axis', 'value'),
     Output('chart-type', 'value'),
     Output('column-filter', 'value'),
     Output('value-filter', 'value'),
     Output('what-if-column', 'value'),
     Output('what-if-adjust', 'value'),
     Output('x-axis-filter-values', 'value')],
    [Input('clear-selection', 'n_clicks')]
)
def clear_selection(n_clicks):
    if n_clicks:
        return None, [], 'bar', None, [], None, 0, []
    return dash.no_update

# Callback to update summary statistics
@app.callback(
    Output('summary-stats', 'children'),
    [Input('filtered-data', 'data')]
)
def update_summary_stats(data):
    if not data:
        return "Please upload data"
    try:
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
    except Exception as e:
        logger.error(f"Error updating summary stats: {e}")
        return f"Error generating summary statistics: {str(e)}"

# Callback to export executive summary
@app.callback(
    Output('download-report', 'data'),
    [Input('export-report', 'n_clicks')],
    [State('filtered-data', 'data'),
     State('kpi-cards', 'children'),
     State('smart-insights', 'children')],
    prevent_initial_call=True
)
def export_executive_summary(n_clicks, data, kpi_cards, smart_insights):
    if not data:
        return None
    df = pd.DataFrame(data)
    return generate_executive_summary(df, kpi_cards, smart_insights)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)