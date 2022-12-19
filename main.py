import numpy as np
import tkinter as tk
# from tkinter.filedialog import askopenfile
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

from algorithms.rnn import *
from algorithms.lstm import *
from algorithms.sarimax import *
from algorithms.xgboost import *

# activate interactive mode, so algorithms shouldn't get interrupted when plt.show()
# plt.ion()

root = tk.Tk()
root.geometry("600x400")
root.title("Energy Consumption Graph")

def run_algorithm():
    if (var1.get() == "LSTM"):
        print("I will now run the " + var1.get() + " algorithm")
        lstm_algorithm()
        print(var1.get() + " algorithm completed")
    elif (var1.get() == "XGBOOST"):
        print("I will now run the " + var1.get() + " algorithm")
        # xgboost_algorithm()
        print(var1.get() + " algorithm completed")
    elif (var1.get() == "SARIMAX"):
        print("I will now run the " + var1.get() + " algorithm")
        sarimax_algorithm()
        print(var1.get() + " algorithm completed")
    elif (var1.get() == "RNN"):
        print("I will now run the " + var1.get() + " algorithm")
        rnn_algorithm()
        print(var1.get() + " algorithm completed")
        
tk.Button(root, text='Run the algorithm', command=run_algorithm).grid(row=2, column=1)

# so root window stays interactive (maybe later)
# tk.Button(root, text='Run the algorithm', command=threading.Thread(target=run_algorithm).start).grid(row=2, column=1)

var1 = tk.StringVar(root, "LSTM")  # Create a variable for strings, and initialize the variable
def printResults():
    print(var1.get())

tk.Radiobutton(root, text="LSTM", variable=var1, value="LSTM", command=printResults).grid(row=4, column=0)
tk.Radiobutton(root, text="XGBOOST", variable=var1, value="XGBOOST", command=printResults).grid(row=5, column=0)
tk.Radiobutton(root, text="SARIMAX", variable=var1, value="SARIMAX", command=printResults).grid(row=6, column=0)
tk.Radiobutton(root, text="RNN", variable=var1, value="RNN", command=printResults).grid(row=7, column=0)

root.mainloop()