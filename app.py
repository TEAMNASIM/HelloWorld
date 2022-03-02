import os
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np
import pickle

# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score
import keras
import altair as alt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History
import random
import string
from pandas.errors import ParserError
import matplotlib.cm as cm

regressor = LogisticRegression()

def main():
	""" Common ML Dataset Explorer """
	st.title("Web App by Nasim Obeid")
	html_temp = """
	<div style="background-color:teal;"><p style="color:white;font-size:50px;padding:10px">Room Occupancy Predictor</p></div>
	"""
	st.markdown(html_temp,unsafe_allow_html=True)

	def file_selector(folder_path='./dataset'):
		filenames = os.listdir(folder_path)
		selected_filename = st.selectbox("Select A file",filenames)
		return os.path.join(folder_path,selected_filename)

	filename = file_selector()
	st.info("You Selected {}".format(filename))

	# Read Data
	df = pd.read_csv(filename)
	# Show Dataset

	if st.checkbox("Show Dataset"):
		st.write(df.astype(str))

	# Show Columns
	if st.button("Column Names"):
		st.write(df.columns)

	# Show Shape
	if st.checkbox("Shape of Dataset"):
		data_dim = st.radio("Show Dimension By ",("Rows","Columns"))
		if data_dim == 'Rows':
			st.text("Number of Rows")
			st.write(df.shape[0])
		elif data_dim == 'Columns':
			st.text("Number of Columns")
			st.write(df.shape[1])
		else:
			st.write(df.shape)

	# Select Columns
	if st.checkbox("Select Columns To Show"):
		all_columns = df.columns.tolist()
		selected_columns = st.multiselect("Select",all_columns)
		new_df = df[selected_columns]
		st.dataframe(new_df)

	# Show Values
	if st.button("Value Counts"):
		st.text("Value Counts By Target/Class")
		st.write(df.iloc[:,-1].value_counts())


	# Show Datatypes
	if st.button("Data Types"):
		st.text(df.dtypes)



	# Show Summary
	if st.checkbox("Summary"):
		st.write(df.describe().T)

	## Plot and Visualization

	st.subheader("Data Visualization")
	# Correlation
	# Seaborn Plot
	if st.checkbox("Correlation Plot[Seaborn]"):
		st.write(sns.heatmap(df.corr(),annot=True))
		st.pyplot()


	# Pie Chart
	if st.checkbox("Pie Plot"):
		all_columns_names = df.columns.tolist()
		if st.button("Generate Pie Plot"):
			st.success("Generating A Pie Plot")
			st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
			st.pyplot()

	# Count Plot
	if st.checkbox("Plot of Value Counts"):
		st.text("Value Counts By Target")
		all_columns_names = df.columns.tolist()
		primary_col = st.selectbox("Primary Columm to GroupBy",all_columns_names)
		selected_columns_names = st.multiselect("Select Columns",all_columns_names)
		if st.button("Plot"):
			st.text("Generate Plot")
			if selected_columns_names:
				vc_plot = df.groupby(primary_col)[selected_columns_names].count()
			else:
				vc_plot = df.iloc[:,-1].value_counts()
			st.write(vc_plot.plot(kind="bar"))
			st.pyplot()


	# Customizable Plot

	st.subheader("Customizable Plot")
	all_columns_names = df.columns.tolist()
	type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
	selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

	if st.button("Generate Plot"):
		st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

		# Plot By Streamlit
		if type_of_plot == 'area':
			cust_data = df[selected_columns_names]
			st.area_chart(cust_data)

		elif type_of_plot == 'bar':
			cust_data = df[selected_columns_names]
			st.bar_chart(cust_data)

		elif type_of_plot == 'line':
			cust_data = df[selected_columns_names]
			st.line_chart(cust_data)

		# Custom Plot
		elif type_of_plot:
			cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
			st.write(cust_plot)
			st.pyplot()

	#Evaluate Model
	st.subheader("Model, Deployment, and Evaluation")
	# Impute nans with mean for numeris and most frequent for categoricals
	cat_imp = SimpleImputer(strategy="most_frequent")
	if len(df.loc[:,df.dtypes == 'object'].columns) != 0:
		df.loc[:,df.dtypes == 'object'] = cat_imp.fit_transform(df.loc[:,df.dtypes == 'object'])
		imp = SimpleImputer(missing_values = np.nan, strategy="mean")
		df.loc[:,df.dtypes != 'object'] = imp.fit_transform(df.loc[:,df.dtypes != 'object'])

# One hot encoding for categorical variables

	features = st.multiselect('select features and target variable',df.columns.tolist())

	cats = df.dtypes == 'object'
	le = LabelEncoder()
	for x in df.columns[cats]:
		df.loc[:,x] = le.fit_transform(df[x])
		onehotencoder = OneHotEncoder()
		df.loc[:,~cats].join(pd.DataFrame(data=onehotencoder.
		fit_transform (df.loc[:,cats]).toarray(), columns=
	onehotencoder.get_feature_names()))

	chosen_target = st.sidebar.selectbox("Please choose target column", (df.columns))

	X = df.loc[:, df.columns != chosen_target]
	scaler = MinMaxScaler(feature_range=(0,1))

	scaler.fit(X)

	X = pd.DataFrame(scaler.transform(X))
	X.columns = df.loc[:, df.columns != chosen_target].columns
	y = df[chosen_target]
# Train test
	X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)

	type = st.sidebar.selectbox("Algorithm type", ("Classification", "Classification"))
	if type == "Classification":
		chosen_classifier = st.sidebar.selectbox("Please choose a classifier", ('Logistic Regression', 'Naive Bayes'))
		if chosen_classifier == 'Logistic Regression':
			max_iter = st.sidebar.slider('max iterations', 1, 100, 10)
	if type == "Classification":
		if chosen_classifier == 'Logistic Regression':
			alg = LogisticRegression()
			model = alg.fit(X_train, y_train)
			predictions = alg.predict(X_test)
			predictions_train = alg.predict(X_train)
		elif chosen_classifier=='Naive Bayes':
			alg = GaussianNB()
			model = alg.fit(X_train, y_train)
			predictions = alg.predict(X_test)
			predictions_train = alg.predict(X_train)


	error_metrics = {}
	if  type == 'Classification':
		error_metrics['Accuracy_test'] = accuracy_score(y_test, predictions)
		error_metrics['Accuracy_train'] = accuracy_score(y_train, predictions_train)
		st.write('### Accuracy Train: ' + str(round(error_metrics['Accuracy_train'], 3)) +
				' -- Accuracy Test: ' +  str(round(error_metrics['Accuracy_test'], 3)))




	if st.button("Thanks"):
		st.balloons()

	st.sidebar.header("About App")
	st.sidebar.info("A Simple ML App for predicting Room Occupancy")


	st.sidebar.header("Developer")
	st.sidebar.info("Nasim Obeid")
	st.sidebar.text("Built with Streamlit")


if __name__ == '__main__':
	main()
