import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create markdown cell
markdown_cell = nbf.v4.new_markdown_cell('''# 3D Tracking Visualization

This notebook demonstrates the 3D tracking visualization from our mono_to_3d project.
We'll visualize the 3D tracking results using matplotlib.''')

# Create code cell
code_cell = nbf.v4.new_code_cell('''%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('.')
from simple_3d_tracker import Simple3DTracker

# Create a tracker instance
tracker = Simple3DTracker()

# Generate sample 3D points
num_points = 50
t = np.linspace(0, 10, num_points)
x = 2 * np.cos(t)
y = 2 * np.sin(t)
z = 0.5 * t

# Create the 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D trajectory
ax.plot(x, y, z, 'b-', label='Tracked Path')
ax.scatter(x, y, z, c='r', marker='o', s=50)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Tracking Visualization')

# Add legend
ax.legend()

# Show the plot
plt.show()''')

# Add cells to notebook
nb.cells = [markdown_cell, code_cell]

# Write the notebook to a file
with open('3d_tracker_visualization.ipynb', 'w') as f:
    nbf.write(nb, f) 