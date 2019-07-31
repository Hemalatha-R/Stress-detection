# -*- coding: utf-8 -*-
"""
Created on Wed May  1 06:31:20 2019

@author: yasin
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from tkinter import ttk

root = Tk()
root.title('STRESS DETECTOR USING ML')
root.geometry('850x650')
root.configure(background="Powderblue")

var = StringVar()
label = Label( root, textvariable = var,font=('arial',20,'bold'),bd=20,background="Powderblue")
var.set("STRESS DETECTOR USING Machine Learning")
label.grid(row=0,columnspan=6) 

def train_file():
     root1=Tk()
     root1.title("login page")
     root1.geometry('600x500')
     root1.configure(background="Powderblue")
     def login():
         user = E.get()
         password = E1.get()
         admin_login(user,password)
     L=Label(root1, text = "Username",bd=8,background="Powderblue",height=1,padx=16,pady=16,font=('arial',16,'bold'),width=10,).grid(row = 0,column=0)
     E=Entry(root1)
     E.grid(row = 0, column = 1)
     L1=Label(root1, text = "Password",bd=8,background="Powderblue",height=1,padx=16,pady=16,font=('arial',16,'bold'),width=10,).grid(row = 1,column=0)
     E1=Entry(root1,show="*")
     E1.grid(row = 1, column = 1)
     B1=Button(root1,text="Login",width=4,height=1,command=login,bd=8,background="Powderblue")
     B1.grid(row = 2, column = 1)
     root1.mainloop()

def admin_login(user,password):
     #print(user,password)
     if user == "hema" and password == "hema":
         root3 = Tk()
         root3.title('choose file')
         root3.geometry('600x300')
         root3.configure(background="Powderblue")
         E2=Button(root3,text="Browse file",height=1,padx=16,pady=16,bd=8,font=('arial',16,'bold'),width=10,bg="Powderblue",command=OpenFile_train)
         E2.place(x=200,y=100)
         
         
         root3.mainloop()  
     else:
         root3 = Tk()
         root3.title('ERROR')
         L2 = Label(root3, text = "user name and password is wrong",font=('arial',16,'bold'),fg='red').grid(row = 2)
         root3.mainloop()

def OpenFile_train():
    name = askopenfilename(initialdir="C:/Users/Batman/Documents/Programming/tkinter/",
                           filetypes =(("Excel File", "*.xlsx"),("All Files","*.*")),
                           title = "Choose a file.")
    try:
        with open(name,'r') as UseFile:
          train(name)
    except FileNotFoundError:
         print("No file exists")

def train(filename): 
    from PIL import ImageTk,Image
    global model,model_norm
    df = pd.read_excel(filename, header=None)#'stress_data.xlsx'
    
    df.columns=['Target', 'ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)','RESP(mV)']
    X_train, X_test, y_train, y_test = train_test_split(df[['ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)','RESP(mV)']], df['Target'],
        test_size=0.30, random_state=12345)
    
    # Min-Max Scaling
    
    minmax_scale = preprocessing.MinMaxScaler().fit(df[['ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)', 'RESP(mV)']])
    df_minmax = minmax_scale.transform(df[['ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)','RESP(mV)']])
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(df_minmax, df['Target'],
        test_size=0.30, random_state=12345)
    
    def plot():
        plt.figure(figsize=(8,6))
    
        plt.scatter(df['Hand GSR(mV)'], df['HR(bpm)'],
                color='green', label='input scale', alpha=0.5)
    
        plt.scatter(df_minmax[:,0], df_minmax[:,1],
                color='blue', label='min-max scaled [min=0, max=1]', alpha=0.3)
    
        plt.title('Hand GSR and HR content of the physiological dataset')
        plt.xlabel('Hand GSR')
        plt.ylabel('HR')
        plt.legend(loc='upper left')
        plt.grid()
        
        plt.tight_layout()
        plt.savefig('graph1.png',bbox_inches='tight')
    plot()
    plt.show()
    
    image = Image.open("graph1.png")
    image = image.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image)  
    panel = Label(root, image=img)
    panel.image = img
    panel.grid(row=2,column=1)
    
    # on non-normalized data
    model = DecisionTreeClassifier(max_leaf_nodes=3)
    fit = model.fit(X_train, y_train)
    
    # on normalized data
    model_norm = DecisionTreeClassifier(max_leaf_nodes=3)
    fit_norm = model_norm.fit(X_train_norm, y_train)
    
    pred_train = model.predict(X_train)
    
    pred_test = model.predict(X_test)
    
    print('Accuracy measure for dataset')
    print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))
    
    pred_train_norm = model_norm.predict(X_train_norm)
    
    print('Accuracy measure for normalized dataset')
    
    pred_test_norm = model_norm.predict(X_test_norm)
    
    print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_norm)))
    
    data = "'Accuracy measure for dataset' = {:.2%}\nAccuracy measure for normalized dataset = {:.2%}".format(metrics.accuracy_score(y_test, pred_test),metrics.accuracy_score(y_test, pred_test_norm))
    labelText = StringVar()
    labelText.set(data)
    output = Label(root, textvariable=labelText,width=45, height=6)
    output.grid(row=3,column=1)

def predict():
    
    root10 = Tk()
    root10.title('Predict STRESS DETECTOR')
    root10.geometry('850x650')
    root10.configure(background="Powderblue")
    
    """var = StringVar()
    label = Label( root, textvariable = var,font=('arial',20,'bold'),bd=20,background="Powderblue")
    var.set("Predict STRESS DETECTOR")
    label.grid(row=0,columnspan=6)
    """
    label_1 = ttk.Label(root10, text ='ECG(mV)',font=("Helvetica", 16),background="Powderblue")
    label_1.grid(row=0,column=0)
    
    Entry_1 = Entry(root10)
    Entry_1.grid(row=0,column=1)
    
    label_2 = ttk.Label(root10, text = 'EMG(mV)',font=("Helvetica", 16),background="Powderblue")
    label_2.grid(row=1,column=0)
    
    Entry_2 = Entry(root10)
    Entry_2.grid(row=1,column=1)
    
    label_3 = ttk.Label(root10, text = 'Foot GSR(mV)',font=("Helvetica", 16,),background="Powderblue")
    label_3.grid(row=2,column=0)
    
    Entry_3 = Entry(root10)
    Entry_3.grid(row=2,column=1)
    
    label_4 = ttk.Label(root10, text = 'Hand GSR(mV)' ,font=("Helvetica", 16),background="Powderblue")
    label_4.grid(row=3,column=0)
    
    Entry_4 = Entry(root10)
    Entry_4.grid(row=3,column=1)
    
    label_5 = ttk.Label(root10, text = 'HR(bpm)',font=("Helvetica", 16),background="Powderblue")
    label_5.grid(row=4,column=0)
    
    Entry_5 = Entry(root10)
    Entry_5.grid(row=4,column=1)
    
    label_6 = ttk.Label(root10, text = 'RESP(mV)',font=("Helvetica", 16),background="Powderblue")
    label_6.grid(row=5,column=0)
    
    Entry_6 = Entry(root10)
    Entry_6.grid(row=5,column=1)
    
    global model,labelText
    
    pred = model.predict([[0.001,0.931,5.91,19.773,99.065,35.59]])
    print(pred)
    pred = model.predict([[-0.005,0.49,8.257,9.853,66.142,45.998]])
    print(pred)
    
    pred = model_norm.predict([[0.001,0.931,5.91,19.773,99.065,35.59]])
    print(pred)
    pred = model_norm.predict([[0.005,0.49,8.257,5.853,80.142,45.998]])
    print(pred)
    
    
    def predout():
        global model,labelText
        pred = model.predict([[Entry_1.get(),Entry_2.get(),Entry_3.get(),Entry_4.get(),Entry_5.get(),Entry_6.get()]])
        
        if pred[0] == 1:
            data = "Stressed"
            output.delete(0, END)
            output.insert(0,data)
        else:
            data = "Not stressed" 
            output.delete(0, END)
            output.insert(0,data)
        
        labelText = StringVar()
        labelText.set(data)
        
        
    label_7 = Button(root10, text = 'output',font=("Helvetica", 16),background="Powderblue",command = predout)
    label_7.grid(row=6,column=0)
    

    output = Entry(root10)
    output.grid(row=6,column=1)
    

    
B = Button(root, text = "Train",height=1,padx=16,pady=16,bd=8,font=('arial',16,'bold'),width=10,bg="Powderblue",command=train_file)
B.grid(row=1,column=0)

B1 = Button(root, text = "Predict",height=1,padx=16,pady=16,bd=8,font=('arial',16,'bold'),width=10,bg="Powderblue",command=predict)
B1.grid(row=1,column=4)

root.mainloop()