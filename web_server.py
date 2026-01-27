from flask import Flask, render_template_string, jsonify
import numpy as np
from simple_3d_tracker import (
    generate_synthetic_tracks, 
    triangulate_tracks,
    set_up_cameras,
    create_interactive_visualization
)
import plotly.io as pio
import logging
import traceback
import argparse
import sys
import os

# Set up argument parser
parser = argparse.ArgumentParser(description='Run the 3D tracker visualization web server')
parser.add_argument('--port', type=int, default=5001, help='Port to run the server on')

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Also log to stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Parse arguments before creating the Flask app
args = parser.parse_args()
port = args.port

logger.info(f"Starting server with port {port}")

# Create Flask app
app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>3D Tracker Visualization</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .error {
            color: #721c24;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>3D Tracker Visualization</h1>
        {% if error %}
        <div class="error">
            <h3>Error:</h3>
            <pre>{{ error }}</pre>
        </div>
        {% else %}
        {{ plot_div | safe }}
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/')
def home():
    try:
        logger.info("Generating visualization...")
        
        # Generate synthetic tracks
        sensor1_track, sensor2_track, original_3d = generate_synthetic_tracks()
        logger.info("Generated synthetic tracks")
        
        # Get camera matrices for triangulation
        P1, P2, _, _ = set_up_cameras()
        
        # Triangulate to get reconstructed 3D points
        reconstructed_3d = triangulate_tracks(sensor1_track, sensor2_track, P1, P2)
        logger.info("Triangulated 3D points")
        
        # Create interactive visualization
        fig = create_interactive_visualization(original_3d, reconstructed_3d, sensor1_track, sensor2_track, (1280, 720))
        logger.info("Created interactive visualization")
        
        if fig is None:
            logger.error("create_interactive_visualization returned None")
            return render_template_string(HTML_TEMPLATE, plot_div=None, error="Visualization creation failed - figure is None")
        
        # Convert to HTML
        plot_div = pio.to_html(fig, full_html=False, include_plotlyjs=True)
        logger.info("Converted plot to HTML")
        
        return render_template_string(HTML_TEMPLATE, plot_div=plot_div, error=None)
    
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template_string(HTML_TEMPLATE, plot_div=None, error=traceback.format_exc())

if __name__ == '__main__':
    # Disable Flask's reloader to avoid port conflicts
    app.config['DEBUG'] = True
    app.config['TEMPLATES_AUTO_RELOAD'] = False
    app.config['USE_RELOADER'] = False
    
    # Run the app with the specified port
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False) 