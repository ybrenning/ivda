import plotly.express as px
import csv
import pandas as pd


df = pd.read_csv('ivda/01/data/Aufgabe-1.csv', on_bad_lines='skip')
px.line(df,x="'Age'", y="'Wage(in Euro)'")
