import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
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
root.geometry("800x500")
label = tk.Label(root, text="Energy Consumption Forecast",
		 fg = "light green",
		 bg = "dark green",
		 font = "Helvetica 16 bold italic")
label.grid(row=1, column=2)
subtitle= tk.Label(root, text="To train a model with your data, please select an algorithm and upload your data as CSV-file",
		 font = "Helvetica 10 bold italic")
subtitle.grid(row=2, column=2)

def run_algorithm():
    #print(checked_columns[0].get())  the checked boxes can be accessed like this, 1 = > checked / 0 => not checked
    if (var1.get() == "LSTM"):
        print("I will now run the " + var1.get() + " algorithm")
        lstm_algorithm(file, checked_columns, col_names)
        print(var1.get() + " algorithm completed")
    elif (var1.get() == "XGBOOST"):
        print("I will now run the " + var1.get() + " algorithm")
        xgboost_algorithm(file, checked_columns, col_names)
        print(var1.get() + " algorithm completed")
    elif (var1.get() == "SARIMAX"):
        print("I will now run the " + var1.get() + " algorithm")
        sarimax_algorithm(file, checked_columns, col_names)
        print(var1.get() + " algorithm completed")
    elif (var1.get() == "RNN"):
        print("I will now run the " + var1.get() + " algorithm")
        rnn_algorithm(file, checked_columns, col_names)
        print(var1.get() + " algorithm completed")

# function to open a new window
# on a button click
def open_new_window():
    # Toplevel object which will
    # be treated as a new window
    new_window = tk.Toplevel(root)

    # sets the title of the
    # Toplevel widget
    new_window.title("Data selection")

    # sets the geometry of toplevel
    new_window.geometry("800x500")
    column_selection_text = tk.Label(new_window,
                        text="Select Columns",
                        font="Helvetica 10 bold italic")
    column_selection_text.grid(row=2, column=0)
    global checked_columns
    checked_columns = []
    row_counter = 2
    for col_name in col_names:
        checked_columns.append(tk.IntVar())
        row_counter+=1
        tk.Checkbutton(new_window, text = col_name, variable = checked_columns[-1],
                 onvalue = 1, offvalue = 0, height=1,
                 width = 20, anchor=tk.W).grid(row=row_counter, column=1)

    tk.Button(new_window, text='Train Model', command=run_algorithm).grid(row=8, column=2)


tk.Button(root, text='Train Model', command=open_new_window).grid(row=8, column=2)

####CSV File Upload

def upload_file():
    f_types = [('CSV files',"*.csv"),('All',"*.*")]
    global file
    file = filedialog.askopenfilename(filetypes=f_types)
    #l1.config(text=file) # display the path
    df = pd.read_csv(file)  # create DataFrame
    global col_names
    col_names = list(df.columns)
    print(col_names)
    label1['text'] = file.split('/')[len(file.split('/'))-1]        #display filename

b1 = tk.Button(root, text='Upload CSV-File',
   width=20,command = lambda:upload_file())
b1.grid(row= 5, column=3)
label1 = tk.Label(text='Please choose a file')
label1.grid(row=6, column=3)

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