import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfile
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


root = tk.Tk()
root.geometry("600x400")
root.title("Energy Consumption Graph")

# read csv files
def open_file():
    file = askopenfile(mode='r', filetypes=[('csv files', '*.csv')])
    if file is not None:
        data = np.random.normal(2000, 23, 5000)
        plt.hist(data, 200)
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.show()
        # fig.canvas.draw()


tk.Button(root, text='Open', command=open_file).grid(row=2, column=1)

def _quit():
    root.quit()
    root.destroy()  # correction: to clean the window when exit

# add a quit button below 'update' button
tk.Button(root, text='Quit', command=_quit).grid(row=2, column=0)

tk.Label(root, text="Choose the desired Algorithm ").grid(row=3, column=0)

def printResults():
    print(var1.get())

var1 = tk.StringVar(root, "LSTM")  # Create a variable for strings, and initialize the variable
tk.Radiobutton(root, text="LSTM", variable=var1, value="LSTM", command=printResults).grid(row=4, column=0)
tk.Radiobutton(root, text="XGBOOST", variable=var1, value="XGBOOST", command=printResults).grid(row=5, column=0)
tk.Radiobutton(root, text="SARIMAX", variable=var1, value="SARIMAX", command=printResults).grid(row=6, column=0)
tk.Radiobutton(root, text="RNN", variable=var1, value="RNN", command=printResults).grid(row=7, column=0)



root.mainloop()