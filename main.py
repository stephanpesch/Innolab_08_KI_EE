import tkinter as tk
import matplotlib.pyplot as plt
import pandas as pd

# Create the main window
window = tk.Tk()
window.title("Graph Plotter")

# Create a function to plot the graph
def plot_graph():
    # Read the CSV file
    df = pd.read_csv("data.csv")

    # Extract the X and Y data
    x = df['X']
    y = df['Y']

    # Plot the graph
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Graph')
    plt.show()

# Create a button to trigger the graph plot
button = tk.Button(text="Plot Graph", command=plot_graph)
button.pack()

# Run the main loop
window.mainloop()
