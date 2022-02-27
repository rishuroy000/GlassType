import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    glass_df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    glass_df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in glass_df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        glass_df.rename(columns_dict, axis = 1, inplace = True)
    return glass_df

g_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = g_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = g_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

@st.cache()
def prediction(model,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe):
  glass_type=model.predict(([model,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe]))
  if glass_type==1:
    return "building windows float processed"
  elif glass_type==2:
    return "building windows non float processed"
  elif glass_type==3:
    return "vehicle windows float processed"
  elif glass_type==4:
    return "vehicle windows non float processed"
  elif glass_type==5:
    return "containers"
  elif glass_type==6:
    return "tableware"
  else:
    return "headlamp"
    
st.title('Glass Type Predictor')
st.sidebar.title('Exploratory Data Analysis')

if st.sidebar.checkbox('Show Raw Data'):
  st.subheader('Full Dataset')
  st.dataframe(g_df)


st.sidebar.subheader("Histogram")

# Choosing features for histograms.
hist_features = st.sidebar.multiselect("Select features to create histograms:", 
                                            ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
# Create histograms.
for feature in hist_features:
    st.subheader(f"Histogram for {feature}")
    plt.figure(figsize = (12, 6))
    plt.hist(g_df[feature], bins = 'sturges', edgecolor = 'black')
    st.pyplot() 

# Create box plots for all the columns.
# Sidebar for box plots.
st.sidebar.subheader("Box Plot")

# Choosing columns for box plots.
box_plot_cols = st.sidebar.multiselect("Select the columns to create box plots:",
                                            ('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType'))

# Create box plots.
for col in box_plot_cols:
    st.subheader(f"Box plot for {col}")
    plt.figure(figsize = (12, 2))
    sns.boxplot(g_df[col])
    st.pyplot()

st.sidebar.subheader("Scatter Plot")
# Remove deprecation warning.
st.set_option('deprecation.showPyplotGlobalUse',False)
# Choosing x-axis values for scatter plots.
features_list = st.sidebar.multiselect("Select the x-axis values:",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
# Creating scatter plots.
for feature in features_list:
    st.subheader(f"Scatter plot between {feature} and GlassType")
    plt.figure(figsize = (12, 6))
    sns.scatterplot(x = feature, y = 'GlassType', data = g_df)
    st.pyplot()
# Remove the code blocks for histogram and box plots.

# Add a subheader in the sidebar with label "Visualisation Selector"
st.sidebar.subheader("Visualisation Selector")
# Add a multiselect in the sidebar with label 'Select the Charts/Plots:'
# and with 6 options passed as a tuple ('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot').
# Store the current value of this widget in a variable 'plot_types'.
plot_types=st.sidebar.multiselect("Select the charts or plots",('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))

if "Histogram" in plot_types:
  st.subheader("Histogram")
  hist_box=st.sidebar.selectbox("Select the columns to create its histogram",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  plt.figure(figsize = (12, 6))
  plt.hist(g_df[feature], bins = 'sturges', edgecolor = 'black')
  st.pyplot() 

if "Box Plot" in plot_types:
  st.subheader("Box Plot")
  boxplot_box=st.sidebar.selectbox("Select the columns to create its boxplot",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  plt.figure(figsize = (12, 2))
  sns.boxplot(g_df[col])
  st.pyplot() 

if "Count Plot" in plot_types:
  st.subheader("Count Plot")
  countplot_box=st.sidebar.selectbox("Select the columns to create its count plot",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  plt.figure(figsize = (12, 2))
  sns.countplot(x='GlassType',data=g_df)
  st.pyplot()
# Create pie chart using the 'matplotlib.pyplot' module and the 'st.pyplot()' function.   
if "Pie Chart" in plot_types:
  st.subheader("Pie Chart")
  plt.figure(figsize = (5,5),dpi=96)
  pie_data=g_df['GlassType'].value_counts()
  plt.pie(pie_data,labels=pie_data.index,autopct='%1.2f%%',startangle=30,explode=np.linspace(0.06,0.16,6))
  st.pyplot()
# Display correlation heatmap using the 'seaborn' module and the 'st.pyplot()' function.
if "Correlation Heatmap" in plot_types:
  st.subheader("Correlation Heatmap")
  plt.figure(figsize = (12, 2))
  sns.heatmap(g_df.corr())
  st.pyplot()
# Display pair plots using the the 'seaborn' module and the 'st.pyplot()' function. 
if "Pair Plot" in plot_types:
  st.subheader("Pair Plot")
  plt.figure(figsize = (12, 2))
  sns.pairplot(g_df)
  st.pyplot()

  # S2.1: Add 9 slider widgets for accepting user input for 9 features.
st.sidebar.subheader('Select Your Values')
ri=st.sidebar.slider('Input RI',float(g_df['RI'].min()),float(g_df['RI'].max()))
na=st.sidebar.slider('Input Na',float(g_df['Na'].min()),float(g_df['Na'].max()))
mg=st.sidebar.slider('Input Mg',float(g_df['Mg'].min()),float(g_df['Mg'].max()))
al=st.sidebar.slider('Input Al',float(g_df['Al'].min()),float(g_df['Al'].max()))
si=st.sidebar.slider('Input Si',float(g_df['Si'].min()),float(g_df['Si'].max()))
k=st.sidebar.slider('Input K',float(g_df['K'].min()),float(g_df['K'].max()))
ca=st.sidebar.slider('Input Ca',float(g_df['Ca'].min()),float(g_df['Ca'].max()))
ba=st.sidebar.slider('Input Ba',float(g_df['Ba'].min()),float(g_df['Ba'].max()))
fe=st.sidebar.slider('Input Fe',float(g_df['Fe'].min()),float(g_df['Fe'].max()))

# S3.1: Add a subheader and multiselect widget.
# Add a subheader in the sidebar with label "Choose Classifier"
st.sidebar.subheader('Choose Classifier')
# Add a selectbox in the sidebar with label 'Classifier'.
# and with 3 options passed as a tuple ('Support Vector Machine', 'Random Forest Classifier').
classifier=st.sidebar.selectbox('Classifier',('Support Vector Machine', 'Random Forest Classifier'))
# Store the current value of this slider in a variable 'classifier'.
if classifier=='Support Vector Machine':
  st.sidebar.subheader('Model Hyperparameters')
  error_rate=st.sidebar.number_input('c (Error Rate)',1,100,step=1)
  kernel=st.sidebar.radio('Kernel',('rbf','linear','poly'))
  gamma=st.sidebar.number_input('Gamma',1,100,step=1)
  if st.sidebar.button('Classify'):
    st.subheader('Support Vector Machine')
    svc_model=SVC(C=error_rate,kernel=kernel,gamma=gamma)
    svc_model.fit(X_train,y_train)
    y_pred = svc_model.predict(X_test)
    accuracy = svc_model.score(X_test, y_test)
    glass_type = prediction(svc_model, ri, na, mg, al, si, k, ca, ba, fe)
    st.write("The Type of glass predicted is:", glass_type)
    st.write("Accuracy", accuracy.round(2))
    plot_confusion_matrix(svc_model, X_test, y_test)
    st.pyplot()

# S5.1: Implement Random Forest Classifier with hyperparameter tuning.
# if classifier == 'Random Forest Classifier', ask user to input the values of 'n_estimators' and 'max_depth'.
if classifier == 'Random Forest Classifier':
  st.sidebar.subheader("Model Hyperparameters")
  n_estimators_input = st.sidebar.number_input("Number of trees in the forest", 100, 5000, step = 10)
  max_depth_input = st.sidebar.number_input("Maximum depth of the tree", 1, 100, step = 1)
  # If the user clicks 'Classify' button, perform prediction and display accuracy score and confusion matrix.
  # This 'if' statement must be inside the above 'if' statement. 
  if st.sidebar.button('Classify'):
    st.subheader("Random Forest Classifier")
    rf_clf= RandomForestClassifier(n_estimators = n_estimators_input, max_depth = max_depth_input, n_jobs = -1)
    rf_clf.fit(X_train,y_train)
    accuracy = rf_clf.score(X_test, y_test)
    glass_type = prediction(rf_clf, ri, na, mg, al, si, k, ca, ba, fe)
    st.write("The Type of glass predicted is:", glass_type)
    st.write("Accuracy", accuracy.round(2))
    plot_confusion_matrix(rf_clf, X_test, y_test)
    st.pyplot()