import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
import os
from PIL import Image

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True, 
               external_stylesheets=['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'])
server = app.server

# Sample data - using PNG files in assets folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(os.getcwd(), 'assets')

YEARS = ['2013', '2017', '2021', '2024', '2034_predicted']
LULC_FILES = {
    '2013': os.path.join(ASSETS_DIR, 'lulc_2013.png'),
    '2017': os.path.join(ASSETS_DIR, 'lulc_2017.png'),
    '2021': os.path.join(ASSETS_DIR, 'lulc_2021.png'),
    '2024': os.path.join(ASSETS_DIR, 'lulc_2024.png'),
    '2034_predicted': os.path.join(ASSETS_DIR, 'LULC_2034_predicted.png')
}

# Verify file existence
for year, path in LULC_FILES.items():
    if not os.path.exists(path):
        print(f"Warning: File for {year} not found at {path}")

# Color mapping and classes
LULC_COLORS = {
    1: '#1f77b4',   
    2: '#2ca02c',   
    3: '#d62728',   
    4: '#ff7f0e',  
    5: '#8c564b'   
}

LULC_CLASSES = {
    1: 'Waterbodies',
    2: 'Vegetation',
    3: 'Built-up',
    4: 'Cropland',
    5: 'Barrenland'
}

# Load area statistics from CSV
def load_area_stats():
    try:
        stats_path = os.path.join(ASSETS_DIR, 'warangal_area_statistics.csv')
        if os.path.exists(stats_path):
            df = pd.read_csv(stats_path)
            
            # Convert from long to wide format
            wide_df = df.pivot(index='Year', columns='Class', values='Area_ha')
            
            # Convert hectares to square kilometers (1 km² = 100 ha)
            wide_df = wide_df / 100
            
            # Rename columns to match our LULC_CLASSES if needed
            column_mapping = {
                'Urban': 'Built-up',
            }
            wide_df = wide_df.rename(columns=column_mapping)
            
            # Verify the index is unique
            if not wide_df.index.is_unique:
                print("Warning: Duplicate years found in area statistics. Using first occurrence.")
                wide_df = wide_df[~wide_df.index.duplicated(keep='first')]
            
            # Convert index to string for consistency with dropdown values
            wide_df.index = wide_df.index.astype(str)
            
            # Add 2034_predicted if it's not already there (using 2034 data)
            if '2034_predicted' not in wide_df.index and '2034' in wide_df.index:
                wide_df.loc['2034_predicted'] = wide_df.loc['2034']
            
            return wide_df.to_dict('index')
        else:
            print(f"Warning: area_statistics.csv not found at {stats_path}")
            # Return sample data if file not found
            return {
                '2013': {'Waterbodies': 326.33, 'Vegetation': 5512.22, 'Built-up': 2209.34, 'Cropland': 2141.68, 'Barrenland': 2705.25},
                '2017': {'Waterbodies': 237.93, 'Vegetation': 3175.73, 'Built-up': 4749.49, 'Cropland': 1736.77, 'Barrenland': 2994.91},
                '2021': {'Waterbodies': 179.27, 'Vegetation': 2975.18, 'Built-up': 5485.32, 'Cropland': 1708.20, 'Barrenland': 2546.86},
                '2024': {'Waterbodies': 159.97, 'Vegetation': 2908.99, 'Built-up': 5548.66, 'Cropland': 1658.10, 'Barrenland': 2619.11},
                '2034': {'Waterbodies': 114.96, 'Vegetation': 438.67, 'Built-up': 6955.65, 'Cropland': 912.89, 'Barrenland': 4291.67},
                '2034_predicted': {'Waterbodies': 114.96, 'Vegetation': 438.67, 'Built-up': 6955.65, 'Cropland': 912.89, 'Barrenland': 4291.67}
            }
    except Exception as e:
        print(f"Error loading area statistics: {str(e)}")
        # Return sample data if error occurs
        return {
            '2013': {'Waterbodies': 326.33, 'Vegetation': 5512.22, 'Built-up': 2209.34, 'Cropland': 2141.68, 'Barrenland': 2705.25},
            '2017': {'Waterbodies': 237.93, 'Vegetation': 3175.73, 'Built-up': 4749.49, 'Cropland': 1736.77, 'Barrenland': 2994.91},
            '2021': {'Waterbodies': 179.27, 'Vegetation': 2975.18, 'Built-up': 5485.32, 'Cropland': 1708.20, 'Barrenland': 2546.86},
            '2024': {'Waterbodies': 159.97, 'Vegetation': 2908.99, 'Built-up': 5548.66, 'Cropland': 1658.10, 'Barrenland': 2619.11},
            '2034': {'Waterbodies': 114.96, 'Vegetation': 438.67, 'Built-up': 6955.65, 'Cropland': 912.89, 'Barrenland': 4291.67},
            '2034_predicted': {'Waterbodies': 114.96, 'Vegetation': 438.67, 'Built-up': 6955.65, 'Cropland': 912.89, 'Barrenland': 4291.67}
        }

# Load and process the statistics
all_stats = load_area_stats()

# Convert to DataFrame and ensure proper formatting
stats_df = pd.DataFrame.from_dict(all_stats, orient='index').fillna(0)
stats_df.index = stats_df.index.astype(str)

# Ensure the columns match our LULC classes
expected_columns = list(LULC_CLASSES.values())
for col in expected_columns:
    if col not in stats_df.columns:
        stats_df[col] = 0  # Add missing columns with zeros
stats_df = stats_df[expected_columns]  # Reorder columns

if '2034_predicted' not in stats_df.index and '2034' in stats_df.index:
    stats_df.loc['2034_predicted'] = stats_df.loc['2034']

# Update the YEARS list based on the actual data we have
YEARS = sorted(stats_df.index.tolist())

# Sample accuracy assessment data
confusion_matrix = np.array([
    [45783, 12345, 23890, 5671, 1103],       # Waterbodies
    [16021, 872543, 128673, 42188, 51649],   # Vegetation
    [26857, 257340, 2156891, 160843, 102337],# Built-up
    [5190, 63415, 178320, 124067, 29540],    # Cropland
    [9834, 46328, 93412, 29822, 149888]      # Barrenland
])


overall_accuracy = 0.6412 

# Load transition matrix image
def get_transition_matrix_image():
    try:
        transition_path = os.path.join(ASSETS_DIR, 'warangal_transition_matrix_2013_2034.png')
        if os.path.exists(transition_path):
            with open(transition_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            return f'data:image/png;base64,{encoded_image}'
        else:
            print(f"Warning: transition_matrix.png not found at {transition_path}")
            return ''
    except Exception as e:
        print(f"Error loading transition matrix: {str(e)}")
        return ''

# Helper function to create map image
def create_map_image(image_path):
    if not os.path.exists(image_path):
        return ''
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

# Helper function to create confusion matrix image
def create_confusion_matrix_image(matrix):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(LULC_CLASSES))
    plt.xticks(tick_marks, LULC_CLASSES.values(), rotation=45)
    plt.yticks(tick_marks, LULC_CLASSES.values())

    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, format(matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{img_str}'

def create_percentage_cards(selected_year):
    # Handle the case where '2034_predicted' might be stored as '2034'
    lookup_year = '2034' if selected_year == '2034_predicted' and '2034' in stats_df.index else selected_year

    if lookup_year not in stats_df.index:
        return html.Div(f"Data not available for {selected_year}", 
                       style={'textAlign': 'center', 'color': 'red'})

    try:
        # Calculate percentages
        row = stats_df.loc[lookup_year]
        total_area = row.sum()
        percentages = (row / total_area * 100).round(2)
        
        # Create cards for each class
        cards = []
        for class_id, class_name in LULC_CLASSES.items():
            if class_name in percentages:
                card = html.Div(
                    [
                        html.H4(class_name, style={
                            'textAlign': 'center', 
                            'marginBottom': '2px',
                            'marginTop':'0px',
                            'fontSize': '20px'
                        }),
                        html.H3(f"{percentages[class_name]}%", 
                               style={
                                   'textAlign': 'center', 
                                   'color': LULC_COLORS[class_id],
                                   'margin': '2px 0',
                                   'fontSize': '20px'
                               }),
                        html.P(f"{row[class_name]:.2f} sq km", 
                              style={
                                  'textAlign': 'center', 
                                  'fontSize': '20px', 
                                  'marginTop': '0px',
                                  'marginBottom': '2px'
                              })
                    ],
                    style={
                        'width': '180px',
                        'padding': '5px',
                        'borderRadius': '12px',
                        'backgroundColor': '#ffffff',
                        'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.15)',
                        'margin': '2px 0'
                    }
                )
                cards.append(card)
        
        return html.Div(cards, style={
            'display': 'flex', 
            'flexDirection': 'column', 
            'gap': '5px',
            'alignItems': 'center'
        })

    except Exception as e:
        return html.Div(f"Error displaying data: {str(e)}", 
                       style={'textAlign': 'center', 'color': 'red'})

# Sidebar styles
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "box-shadow": "2px 0 5px rgba(0,0,0,0.1)",
    "z-index": 1,
    "transition": "all 0.3s"
}

SIDEBAR_HIDDEN = {
    "position": "fixed",
    "top": 0,
    "left": "-18rem",
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "box-shadow": "2px 0 5px rgba(0,0,0,0.1)",
    "z-index": 1,
    "transition": "all 0.3s"
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "transition": "all 0.3s"
}

CONTENT_STYLE_FULL = {
    "margin-left": "2rem",
    "margin-right": "0rem",
    "padding": "2rem 1rem",
    "transition": "all 0.3s"
}

# Sidebar navigation links (without toggle button)
sidebar = html.Div(
    [
        html.H2("Warangal Dashboard", className="display-4", style={'fontSize': '24px'}),
        html.Hr(),
        html.P(
            "Land Use/Land Cover Analysis", className="lead",
            style={'fontSize': '16px', 'marginBottom': '30px'}
        ),
        dcc.Link(
            "Map Viewer & Statistics",
            href="/",
            id="map-viewer-link",
            style={
                'display': 'block',
                'padding': '10px',
                'margin': '5px 0',
                'borderRadius': '12px',
                'backgroundColor': '#e9ecef',
                'color': '#495057',
                'textDecoration': 'none',
                'fontWeight': 'bold'
            }
        ),
        dcc.Link(
            "Change Detection",
            href="/change-detection",
            id="change-detection-link",
            style={
                'display': 'block',
                'padding': '10px',
                'margin': '5px 0',
                'borderRadius': '12px',
                'backgroundColor': '#e9ecef',
                'color': '#495057',
                'textDecoration': 'none',
                'fontWeight': 'bold'
            }
        ),
        dcc.Link(
            "Accuracy Assessment",
            href="/accuracy-assessment",
            id="accuracy-assessment-link",
            style={
                'display': 'block',
                'padding': '10px',
                'margin': '5px 0',
                'borderRadius': '12px',
                'backgroundColor': '#e9ecef',
                'color': '#495057',
                'textDecoration': 'none',
                'fontWeight': 'bold'
            }
        ),
    ],
    id="sidebar",
    style=SIDEBAR_HIDDEN,  # Start with sidebar hidden
)

# Main content layout with single toggle button
content = html.Div([
    # Global toggle button (fixed position)
    html.Div([
        html.Button(
            html.I(className="fas fa-bars"), 
            id="sidebar-toggle",
            n_clicks=0,
            style={
                'position': 'fixed',
                'left': '10px',
                'top': '10px',
                'zIndex': 1000,
                'background': 'none',
                'border': 'none',
                'fontSize': '20px',
                'cursor': 'pointer'
            }
        )
    ]),
    # Page content
    html.Div(id="page-content", style=CONTENT_STYLE_FULL)
])

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content,
    dcc.Store(id='sidebar-state', data='hidden')
])

# Map Viewer & Statistics Page

map_viewer_layout = html.Div([
    # Row 1: Dropdowns
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='year-selector',
                options=[
                    {'label': '2013', 'value': '2013'},
                    {'label': '2017', 'value': '2017'},
                    {'label': '2021', 'value': '2021'},
                    {'label': '2024', 'value': '2024'},
                    {'label': '2034 Predicted', 'value': '2034_predicted'}
                ],
                value='2013',
                clearable=False
            )
        ], style={'width': '25%', 'display': 'inline-block', 'padding': '0px'}),

        html.Div([
            dcc.Dropdown(
                id='class-selector',
                options=[{'label': cls, 'value': cls} for cls in LULC_CLASSES.values()],
                value=None,
                multi=True,
            )
        ], style={'width': '25%', 'display': 'inline-block', 'padding': '0px'})
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'paddingTop': '0px'}),

    # Row 2: Map, Line Chart and Cards
    html.Div([
        # Map Container
        html.Div([
            html.H4("", style={'textAlign': 'center'}),
            html.Div(id='map-container', style={'height': '550px'})
        ], style={'width': '100%', 'padding': '0px', 'display': 'inline-block','margin':'0px'}),
        
        # Line Chart
        html.Div([
            
            dcc.Graph(id='area-trend-chart', style={'height': '500px'})
        ], style={'width': '100%', 'padding': '10px', 'display': 'inline-block','margin':'0px'}),
        
        # Percentage Cards
        html.Div([
            
            html.Div(id='percentage-cards',
                    style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'gap': '10px',
                        'alignItems': 'center'
                    }
            )
        ], style={
            'width': '20%', 
            'padding': '5px', 
            'display': 'inline-block',
            'verticalAlign': 'top'
        })
    ], style={'display': 'flex', 'width': '100%', 'marginBottom': '0px'}),

    # Row 3: Bar Chart
    html.Div([
        dcc.Graph(
            figure=px.bar(
                stats_df[~stats_df.index.astype(str).str.contains('2034_predicted')]  # Filter out 2034_predicted
                .reset_index()
                .melt(id_vars='index', var_name='Class', value_name='Area'),
                x='index', y='Area', color='Class', barmode='group',
                color_discrete_map={v: LULC_COLORS[k] for k, v in LULC_CLASSES.items()}
            ).update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#2c3e50'},
                showlegend=True,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title='Year',
                yaxis_title='Area (sq.km)',
                yaxis=dict(
                    visible=True,
                    showline=True,
                    linecolor='#d3d3d3',  
                    linewidth=2
                ),
                xaxis=dict(
                    visible=True,
                    showline=True,
                    linecolor='#d3d3d3',  
                    linewidth=2
                ),
            ),
            style={'height': '400px'}
        )
    ], style={'width': '100%', 'padding': '0px', 'margin': '0px'})



])
# Change Detection Page
change_detection_layout = html.Div([
    html.H3("Change Detection Analysis", style={'textAlign': 'center', 'color': '#27ae60', 'paddingTop': '0px'}),
    
    # Row 1: Before/After Images
    html.Div([
        html.Div([
            html.Label("Before Year:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='before-year',
                options=[{'label': year, 'value': year} for year in YEARS if year != '2034_predicted'],
                value=YEARS[0],
                style={'width': '80%', 'margin': '0 auto'},
                clearable=False
            ),
            html.Img(id='before-image', style={'width': '90%', 'padding': '10px'})
        ], style={'width': '48%', 'display': 'inline-block', 'textAlign': 'center'}),

        html.Div([
            html.Label("After Year:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='after-year',
                options=[{'label': year, 'value': year} for year in YEARS if year in ['2017', '2021', '2024', '2034_predicted']],
                value='2034_predicted',
                style={'width': '80%', 'margin': '0 auto'},
                clearable=False
            ),
            html.Img(id='after-image', style={'width': '90%', 'padding': '0px'})
        ], style={'width': '48%', 'display': 'inline-block', 'textAlign': 'center'})
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    # Row 2: Urban Expansion Bar Chart
    html.Div([
        dcc.Graph(
            id='urban-expansion-barchart',
            style={'height': '500px', 'width': '100%'}
        )
    ], style={'width': '100%', 'padding': '10px', 'marginBottom': '20px'}),
    
    # Row 3: Transition Matrix and Change Graph
    html.Div([
        # Transition Matrix
        html.Div([
            html.Img(src=get_transition_matrix_image(), style={'width': '100%', 'display': 'block', 'margin': '0 auto'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Change Graph
        html.Div([
            dcc.Graph(id='change-graph', style={'height': '600px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginTop': '20px'})
])
# Accuracy Assessment Page
accuracy_assessment_layout = html.Div([
    # Heading with Overall Accuracy
    html.H3(f"Overall Accuracy: {overall_accuracy*100:.2f}%", 
           style={'textAlign': 'center', 'color': '#e74c3c', 'paddingTop': '0px', 'marginBottom': '20px'}),
    
    # Single Row Layout with equal height components
    html.Div([
        # LULC Map (left)
        html.Div([
            html.Img(
                src=create_map_image(LULC_FILES['2024']),
                style={
                    'width': '100%', 
                    'height': '500px',
                    'objectFit': 'contain',
                    'display': 'block',
                    'padding': '0px',
                    'margin': '0px'
                }
            )
        ], style={
            'width': '40%', 
            'display': 'inline-block', 
            'verticalAlign': 'top',
            'height': '500px',
            'padding': '0px',
            'margin': '0px'
        }),
        
        # Accuracy Image (center)
        html.Div([
            html.Img(
                src=app.get_asset_url('accuracy.png'),
                style={
                    'width': '100%', 
                    'height': '500px',
                    'objectFit': 'contain',
                    'display': 'block',
                    'padding': '0px',
                    'margin': '0px'
                }
            )
        ], style={
            'width': '60%', 
            'display': 'inline-block', 
            'verticalAlign': 'top',
            'height': '500px',
            'padding': '0px',
            'margin': '0px'
        }),
        
        # Heatmap (right)
        html.Div([
            dcc.Graph(
                figure=go.Figure(
                    data=go.Heatmap(
                        z=confusion_matrix,
                        x=list(LULC_CLASSES.values()),
                        y=list(LULC_CLASSES.values()),
                        colorscale='Viridis',
                        hoverongaps=False,
                        text=confusion_matrix,
                        texttemplate="%{text}",
                        textfont={"size":12}
                    ),
                    layout=go.Layout(
                        title='Confusion Matrix',
                        xaxis_title='Predicted Class',
                        yaxis_title='Actual Class',
                        height=500,
                        margin=dict(l=0, r=0, t=30, b=0)  # Tightened margins
                    )
                ),
                style={'height': '500px', 'width': '100%'}
            )
        ], style={
            'width': '50%', 
            'display': 'inline-block', 
            'verticalAlign': 'top',
            'height': '530px',  # 500px + padding
            'padding': '0px'
        })
    ], style={
        'display': 'flex', 
        'justifyContent': 'space-between',
        'alignItems': 'flex-start',
        'height': '500px',
        'padding': '0px',
        'margin': '0px',
        'width': '100%'
    })
], style={'padding': '0px', 'margin': '0px'})
# Callback to switch between pages
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        return map_viewer_layout
    elif pathname == "/change-detection":
        return change_detection_layout
    elif pathname == "/accuracy-assessment":
        return accuracy_assessment_layout
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

# Callback for Map Viewer
@app.callback(
    Output('map-container', 'children'),
    Input('year-selector', 'value')
)
def update_map(year):
    lulc_image_path = LULC_FILES.get(year)

    # Check if the file exists
    if lulc_image_path and os.path.exists(lulc_image_path):
        # Open the image using PIL
        img = Image.open(lulc_image_path)
        # Convert the image to a base64 encoded string for embedding in HTML
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_data_uri = f"data:image/png;base64,{img_str}"

        return html.Img(src=img_data_uri, style={'width': '100%', 'display': 'block', 'margin': '0px'})
    else:
        return html.Div([
            f"Image not found for the selected year. Please check the assets folder. Tried path: {lulc_image_path}"
        ])

# Callback for Percentage Cards
@app.callback(
    Output('percentage-cards', 'children'),
    Input('year-selector', 'value')
)
def update_percentage_cards(selected_year):
    print(f"Callback triggered with year: {selected_year}")  # Debug print
    return create_percentage_cards(selected_year)

# Callback for Area Statistics
@app.callback(
    Output('area-trend-chart', 'figure'),
    Input('class-selector', 'value')
)
def update_area_chart(selected_classes):
    # If nothing is selected, use all classes
    if not selected_classes:
        selected_classes = list(LULC_CLASSES.values())

    fig = go.Figure()

    # Filter out '2034_predicted' from the index
    filtered_df = stats_df[~stats_df.index.astype(str).str.contains('2034_predicted')]

    for cls in selected_classes:
        if cls in filtered_df.columns:
            fig.add_trace(go.Scatter(
                x=filtered_df.index,
                y=filtered_df[cls],
                name=cls,
                mode='lines+markers',
                line=dict(width=2),
                marker=dict(size=8),
                hovertemplate='Year: %{x}<br>Area: %{y:.2f} sq km<extra></extra>'
            ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'),
        margin=dict(l=0, r=10, t=20, b=0),
        xaxis=dict(
            title='Year',
            showline=True,
            linecolor='#d3d3d3',
            linewidth=2,
            showgrid=False,
            zeroline=False,
            ticks='outside',
            tickcolor='#d3d3d3',
            tickwidth=2
        ),
        yaxis=dict(
            title='Area (sq km)',
            showline=True,
            linecolor='#d3d3d3',
            linewidth=2,
            showgrid=False,
            zeroline=False,
            ticks='outside',
            tickcolor='#d3d3d3',
            tickwidth=2
        )
    )

    return fig

# Callbacks for Change Detection
@app.callback(
    Output('before-image', 'src'),
    Input('before-year', 'value')
)
def update_before_image(year):
    if not year or year not in LULC_FILES:
        return ''
    try:
        return create_map_image(LULC_FILES[year])
    except Exception as e:
        print(f"Error updating before image: {str(e)}")
        return ''

@app.callback(
    Output('after-image', 'src'),
    Input('after-year', 'value')
)
def update_after_image(year):
    if not year or year not in LULC_FILES:
        return ''
    try:
        return create_map_image(LULC_FILES[year])
    except Exception as e:
        print(f"Error updating after image: {str(e)}")
        return ''

@app.callback(
    Output('change-graph', 'figure'),
    [Input('before-year', 'value'),
     Input('after-year', 'value')]
)
def update_change_graph(before_year, after_year):
    if not before_year or not after_year or before_year not in all_stats or after_year not in all_stats:
        return go.Figure().update_layout(
            title="Please select valid years for comparison",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#2c3e50'}
        )

    try:
        # Calculate changes between years
        before_stats = all_stats[before_year]
        after_stats = all_stats[after_year]
        
        changes = {k: after_stats.get(k, 0) - before_stats.get(k, 0) for k in set(before_stats) | set(after_stats)}
        
        # Create a bar chart of changes with consistent colors
        fig = go.Figure()
        
        # Map class names to their corresponding colors
        class_colors = {
            'Waterbodies': LULC_COLORS[1],
            'Vegetation': LULC_COLORS[2],
            'Built-up': LULC_COLORS[3],
            'Cropland': LULC_COLORS[4],
            'Barrenland': LULC_COLORS[5]
        }
        
        for class_name, change in changes.items():
            fig.add_trace(go.Bar(
                x=[class_name],
                y=[change],
                name=class_name,
                marker_color=class_colors.get(class_name, '#cccccc'),
                text=[f"{change:.2f}"], 
                textposition='auto',     
                hovertemplate='Class: %{x}<br>Change: %{y:.2f} sq km<extra></extra>'
            ))

        fig.update_layout(
            title=f'LULC Change {before_year} to {after_year} ',
            xaxis_title='LULC Class',
            yaxis_title='Area Change (sq km)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#2c3e50'},
            showlegend=False,
            bargap=0.2,
            # Explicit axis styling
            xaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='#d3d3d3',
                ticks='outside',
                showgrid=False
            ),
            yaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='#d3d3d3',
                ticks='outside',
                showgrid=False,

                zeroline=False

            )
        )
        
        return fig
    except Exception as e:
        error_msg = f"Error in change analysis: {str(e)}"
        print(error_msg)
        return go.Figure().update_layout(
            title=error_msg,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#2c3e50'}
        )
        
@app.callback(
    Output('urban-expansion-barchart', 'figure'),
    Input('url', 'pathname')  # Trigger when page loads
)
def update_urban_expansion_barchart(pathname):
    if pathname != "/change-detection":
        return go.Figure()
    
    # Prepare data - group into Urban vs Other
    # First filter out '2034_predicted' if it exists
    years_data = stats_df[stats_df.index != '2034_predicted']
    years = sorted(years_data.index)
    
    # Calculate total area for each year
    total_areas = years_data.sum(axis=1)
    
    # Get built-up (urban) areas
    urban_areas = years_data['Built-up']
    
    # Calculate "Other" areas (total - urban)
    other_areas = total_areas - urban_areas
    
    # Create the figure
    fig = go.Figure()
    
    # Add Urban bars with text
    fig.add_trace(go.Bar(
        x=years,
        y=urban_areas,
        name='Built-up (Urban)',
        marker_color=LULC_COLORS[3],  # Using the red color for urban
        text=urban_areas.round(2),
        textposition='auto',
        texttemplate='%{text:.2f}',
        hovertemplate='Year: %{x}<br>Urban Area: %{y:.2f} sq km<extra></extra>'
    ))
    
    # Add Other bars with text
    fig.add_trace(go.Bar(
        x=years,
        y=other_areas,
        name='Other Land Uses',
        marker_color='#7f7f7f',  # Gray color for other classes
        text=other_areas.round(2),
        textposition='auto',
        texttemplate='%{text:.2f}',
        hovertemplate='Year: %{x}<br>Other Area: %{y:.2f} sq km<extra></extra>'
    ))
    
    fig.update_layout(
        title='Urban Expansion vs Other Land Uses',
        xaxis_title='Year',
        yaxis_title='Area (sq km)',
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#2c3e50', 'size': 12},
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#d3d3d3',
            ticks='outside',
            showgrid=False,
            # Ensure only existing years are shown
            type='category',
            categoryorder='array',
            categoryarray=years
        ),
        yaxis=dict(
            showline=True,
            linewidth=2,
            linecolor='#d3d3d3',
            ticks='outside',
            showgrid=False,
            zeroline=False
        ),
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    
    # Adjust text font size and color
    fig.update_traces(
        textfont_size=12,
        textfont_color='white',
        insidetextfont_color='white',
        outsidetextfont_color='black'
    )
    
    return fig
# Callback for sidebar toggle
@app.callback(
    [Output("sidebar", "style"),
     Output("page-content", "style"),
     Output('sidebar-state', 'data')],
    [Input("sidebar-toggle", "n_clicks")],
    [State('sidebar-state', 'data')]
)
def toggle_sidebar(n_clicks, state):
    if n_clicks is None:
        return SIDEBAR_HIDDEN, CONTENT_STYLE_FULL, 'hidden'
    
    if state == 'hidden':
        return SIDEBAR_STYLE, CONTENT_STYLE, 'visible'
    else:
        return SIDEBAR_HIDDEN, CONTENT_STYLE_FULL, 'hidden'

# Callback to update the active link in the sidebar
@app.callback(
    [Output("map-viewer-link", "style"),
     Output("change-detection-link", "style"),
     Output("accuracy-assessment-link", "style")],
    [Input("url", "pathname")],
    [State('sidebar-state', 'data')]
)
def update_active_link(pathname, sidebar_state):
    # Default style for all links
    default_style = {
        'display': 'block',
        'padding': '10px',
        'margin': '5px 0',
        'borderRadius': '12px',
        'backgroundColor': '#e9ecef',
        'color': '#495057',
        'textDecoration': 'none',
        'fontWeight': 'bold'
    }

    # Active link style
    active_style = default_style.copy()
    active_style['backgroundColor'] = '#27ae60'
    active_style['color'] = 'white'

    # Initialize all links with default style
    map_style = default_style.copy()
    change_style = default_style.copy()
    accuracy_style = default_style.copy()

    # Update the style for the active link
    if pathname == "/":
        map_style = active_style
    elif pathname == "/change-detection":
        change_style = active_style
    elif pathname == "/accuracy-assessment":
        accuracy_style = active_style

    return map_style, change_style, accuracy_style

if __name__ == '__main__':
    app.run(debug=False)