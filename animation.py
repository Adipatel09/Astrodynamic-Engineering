import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

def interpolate_extreme_values(df):
    """
    Interpolate extreme values (>1e9 or <1e-9) using linear interpolation
    """
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Sort by time to ensure proper interpolation
    df = df.sort_values(['time', 'ECEF'])
    
    # Create separate dataframes for x, y, z coordinates
    positions = {}
    for coord in ['x', 'y', 'z']:
        coord_data = df[df['ECEF'] == coord].copy()
        
        # Mark extreme values
        extreme_mask = (abs(coord_data['position']) > 1e9) | (abs(coord_data['position']) < 1e-9)
        coord_data.loc[extreme_mask, 'position'] = np.nan
        
        # Interpolate missing values
        coord_data['position'] = coord_data['position'].interpolate(method='linear')
        
        # If still have NaN at edges, use forward/backward fill
        coord_data['position'] = coord_data['position'].fillna(method='ffill').fillna(method='bfill')
        
        positions[coord] = coord_data
    
    return positions

def create_animation():
    # Read the GPS data
    df = pd.read_csv('GPS meas.csv')
    
    # Get interpolated positions
    positions = interpolate_extreme_values(df)
    
    # Create time series for plotting
    times = sorted(set(pd.to_datetime(df['time'])))
    
    # Create arrays for coordinates
    x_coords = []
    y_coords = []
    z_coords = []
    
    for t in times:
        x_val = positions['x'][positions['x']['time'] == t]['position'].values
        y_val = positions['y'][positions['y']['time'] == t]['position'].values
        z_val = positions['z'][positions['z']['time'] == t]['position'].values
        
        if len(x_val) > 0 and len(y_val) > 0 and len(z_val) > 0:
            x_coords.append(float(x_val[0]))
            y_coords.append(float(y_val[0]))
            z_coords.append(float(z_val[0]))
    
    # Create the animation frames
    frames = []
    for i in range(len(x_coords)):
        frame = go.Frame(
            data=[
                # Historical trajectory
                go.Scatter3d(
                    x=x_coords[:i+1],
                    y=y_coords[:i+1],
                    z=z_coords[:i+1],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=list(range(i+1)),
                        colorscale='Viridis',
                        opacity=0.8
                    )
                ),
                # Current point (bigger and different color)
                go.Scatter3d(
                    x=[x_coords[i]],
                    y=[y_coords[i]],
                    z=[z_coords[i]],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='circle'
                    ),
                    name='Current Position'
                )
            ],
            name=str(i)
        )
        frames.append(frame)
    
    # Create the initial figure
    fig = go.Figure(
        data=[
            # Initial position
            go.Scatter3d(
                x=[x_coords[0]],
                y=[y_coords[0]],
                z=[z_coords[0]],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='circle'
                ),
                name='Current Position'
            )
        ],
        frames=frames
    )
    
    # Add Earth sphere
    r = 6378.137  # Earth radius in km
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(
        go.Surface(
            x=x, y=y, z=z,
            opacity=0.3,
            colorscale=[[0, 'lightblue'], [1, 'lightblue']],
            showscale=False,
            name='Earth'
        )
    )

    # Update layout with animation settings
    fig.update_layout(
        title='GPS Satellite Trajectory',
        scene=dict(
            xaxis_title='X Position (km)',
            yaxis_title='Y Position (km)',
            zaxis_title='Z Position (km)',
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 50, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 0}
                }]
            }, {
                'label': 'Pause',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }]
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Frame: '},
            'steps': [
                {
                    'method': 'animate',
                    'label': str(i),
                    'args': [[str(i)], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
                for i in range(len(frames))
            ]
        }]
    )

    # Save the animation as HTML
    fig.write_html('gps_trajectory_animation.html')
    
    # Show the plot in the browser
    fig.show()

if __name__ == "__main__":
    create_animation() 