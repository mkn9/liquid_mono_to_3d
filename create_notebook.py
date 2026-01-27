import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create markdown cell
markdown_cell = nbf.v4.new_markdown_cell('''# Test Plot Notebook

This notebook demonstrates basic plotting functionality using matplotlib.''')

# Create code cell
code_cell = nbf.v4.new_code_cell('''%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a simple plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='sin(x)')
plt.title('Test Plot: Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.legend()
plt.show()''')

# Add cells to notebook
nb.cells = [markdown_cell, code_cell]

# Write the notebook to a file
nbf.write(nb, 'test_plot.ipynb') 