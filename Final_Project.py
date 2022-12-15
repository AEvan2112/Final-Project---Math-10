import streamlit as st
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import altair as alt
import sklearn
from sklearn.cluster import KMeans

st.title("U.S. Educational Finances Data Analyzer")

st.markdown(
    "Alexander Chandra **https://github.com/AEvan2112?tab=repositories**"
    ,unsafe_allow_html=True)
st.write("40029708")

st.write("Reading the district level csv file and cleaning it from not applicable values (2)")

df_district = pd.read_csv(r"C:\Users\alexa\Downloads\districts-reduced.csv",na_values = " ")
df_district = df_district[df_district.notna().all(axis=1)].copy()

st.write("Checking the shape of the dataframe")

df_district.shape

st.write("Reading the states level csv file and cleaning it from not applicable values")

df_state = pd.read_csv(r"C:\Users\alexa\Downloads\states.csv",na_values = " ")
df_state = df_state[df_state.notna().all(axis=1)].copy()

st.write("Checking the shape of the state level dataframe to ensure that it is compatible with Altair")

df_state.shape

st.write("Gaining the names of the states")

df_states_name = pd.DataFrame()
df_states_name = df_state.STATE[:51].copy()

st.write("Please enter your state choice")

state_choice = st.selectbox("Choose the state",options = df_states_name)
df_dchoice = df_district[df_district["STATE"] == state_choice].copy()

st.write("Since the Altair library can only handle 5000 data, the app only takes the first 5000 rows in the dataframe df_district.")

if len(df_dchoice) > 5000:
    df_dchoice = df_dchoice[:5000].copy()
    
st.write("Begin analyzing the profit or loss of each school in district level in the chosen state")

st.write("The profit or loss is computed by the difference between the total revenues and total expenditures.")

df_dchoice["PROFIT/LOSS"] = df_dchoice.loc[:,"TOTALREV"] - df_dchoice.loc[:,"TOTALEXP"]

st.write("To get which school had the most profit or loss, the dataframe is sorted ascendingly based on the profit or loss.")

sorted_df_dchoice = df_dchoice.sort_values("PROFIT/LOSS",ascending=True)

st.write("Here is the sorted dataframe of the district level.")

sorted_df_dchoice

choice = st.selectbox("Please enter your choice between the most profit or the most loss",options=["PROFIT","LOSS"])

if choice == "PROFIT":
    st.write(f"Based on above, the school in {sorted_df_dchoice.iloc[-1].loc['STATE']} that had the most profit is {sorted_df_dchoice.iloc[-1].loc['NAME']} in the year {sorted_df_dchoice.iloc[-1].loc['YRDATA']} with profit {sorted_df_dchoice.iloc[-1].loc['PROFIT/LOSS']} U.S. dollars.")
else:
    st.write(f"Based on above, the school in {sorted_df_dchoice.iloc[0].loc['STATE']} that had the most loss is {sorted_df_dchoice.iloc[0].loc['NAME']} in the year {sorted_df_dchoice.iloc[0].loc['YRDATA']} with loss {sorted_df_dchoice.iloc[0].loc['PROFIT/LOSS']} U.S. dollars.")

st.write("Begin clustering data based on their revenues and expenditures statistics (3)")

st.write("To gather the revenues and expenditures statistics, the numeric columns are extracted. (2)")

numeric_cols_dist = [c for c in df_dchoice.columns if is_numeric_dtype(df_dchoice[c])]

num_df_dchoice = df_dchoice[numeric_cols_dist].copy()

num_df_dchoice = num_df_dchoice[num_df_dchoice["PROFIT/LOSS"].notna()].copy()

num_df_dchoice

st.write("To begin clustering, the application uses Kmeans to cluster data whose distances are the nearest to some different assigned groups")

num_of_clusters = st.slider("Please choose the number of clusters",1,10)

kmeans = KMeans(num_of_clusters)

kmeans.fit(num_df_dchoice)

df_dchoice["CLUSTER"] = kmeans.predict(num_df_dchoice)

st.write("Here is the chart of the clusters. (4),(7)")

district_chart = alt.Chart(df_dchoice).mark_circle().encode(
    x = "ENROLL",
    y = "PROFIT/LOSS",
    color = alt.Color("CLUSTER:O",scale=alt.Scale(scheme='turbo',reverse=True)),
    tooltip = ["NAME","YRDATA","TOTALREV","TOTALEXP","PROFIT/LOSS","CLUSTER"],
).properties(
    width = 500,
    height = 500,
    title = state_choice + "'s District Level Educational Finances Clustering"
)
    
st.altair_chart(district_chart)

st.write("Reading the state level csv file and cleaning it from not applicable values (2)")

df_schoice = df_state[df_state["STATE"] == state_choice].copy()

df_schoice["PROFIT/LOSS"] = df_schoice.loc[:,"TOTAL_REVENUE"] - df_schoice.loc[:,"TOTAL_EXPENDITURE"]

sorted_df_schoice = df_schoice.sort_values("YEAR",ascending=True)

st.write("Here is the dataframe of the state level dataframe sorted ascendingly based on year (10)")

sorted_df_schoice

st.write("Extracting the numeric columns to style the dataframe (2)")

numeric_cols_stat = [c for c in sorted_df_schoice.columns if is_numeric_dtype(sorted_df_schoice[c])]

num_df_schoice = sorted_df_schoice[numeric_cols_stat].copy()

st.write("Changing the float datatype into int to maintain uniformity. (6)")

num_df_schoice = num_df_schoice.astype(int)

st.write("Changing the font color of negative values to red to signify losses instead of profits (5)")

def style_negative(value, props=''):
    return props if value < 0 else None

styled_df_schoice = num_df_schoice.style.applymap(style_negative, props='color:red;')

styled_df_schoice

st.write("Highlighting the maximum values of each column with blue and minimum values of each column with pink (6)")

def highlight_max(value, props=''):
    return np.where(value == np.nanmax(value.values), props, '')

def highlight_min(value, props=''):
    return np.where(value == np.nanmin(value.values), props, '')

styled2_df_schoice = styled_df_schoice.apply(highlight_max, props='color:white;background-color:blue', axis=0)
styled2_df_schoice = styled_df_schoice.apply(highlight_min, props='color:white;background-color:pink', axis=0)

styled2_df_schoice

st.write("Here is the chart of annual profit or loss changes of the chosen state. (4),(7)")

state_chart = alt.Chart(df_schoice).mark_circle().encode(
    x = alt.X("YEAR",scale=alt.Scale(zero=False)),
    y = "PROFIT/LOSS",
    color = alt.Color("TOTAL_REVENUE",scale=alt.Scale(scheme='accent')),
    size = alt.Size("ENROLL",scale=alt.Scale(zero=False)),
    tooltip = ["YEAR","ENROLL","TOTAL_REVENUE","TOTAL_EXPENDITURE","PROFIT/LOSS"],
).properties(
    width = 500,
    height = 500,
    title = state_choice + "'s Annual Educational Finances"
)

st.altair_chart(state_chart)

st.write("Begin analyzing the trend of annual profit or loss changes through linear regression (8)")

X = np.array(num_df_schoice["YEAR"]).reshape(-1,1) # Input
y = np.array(num_df_schoice["PROFIT/LOSS"]).reshape(-1,1) # Output

model = sklearn.linear_model.LinearRegression()

model.fit(X, y)

st.write(f"The model's coefficient is {model.coef_}.")

st.write(f"The model's intercept is {model.intercept_}.")

if model.coef_ > 0:
    analysis = "increase"
elif model.coef_ < 0:
    analysis = "decrease"
else:
    analysis = "not have significant change"

st.write("Based on the coefficient, the model predicts that the value of profit or loss is likely to " + analysis + " as time continues.")

year_pred = st.slider("Choose the year you want to predict the profit or loss",1980,2080)

X_new = [[year_pred]]

st.write("The output is rounded to 2 decimals to match the currency. (9)")

profit_loss_pred = round(float(model.predict(X_new)),2)

st.write(f"For the year {year_pred}, the predicted profit or loss for the state {state_choice} is {profit_loss_pred} dollars.")

st.write("This linear model may not be the best since the only independent variable is the year.")

st.write("References")

st.write("Link of the datasets (1)")
st.markdown(
    "U.S. Educational Finances - Kaggle **https://www.kaggle.com/noriuk/us-educational-finances**"
    ,unsafe_allow_html=True)

st.write("Link for the data cleaning parts (2)")
st.markdown(
    "Overfitting in Tensorflow - Christopher Davis **https://christopherdavisuci.github.io/UCI-Math-10/Week10/overfitting.html**"
    ,unsafe_allow_html=True)

st.write("Link for the clustering part (3)")
st.markdown(
    "K-Means clustering Using scikit-learn - Christopher Davis **https://christopherdavisuci.github.io/UCI-Math-10/Week7/Week6-Friday.html**"
    ,unsafe_allow_html=True)

st.write("Link for the Altair charts parts (4)")
st.markdown(
    "First Two Examples with Altair - Christopher Davis **https://christopherdavisuci.github.io/UCI-Math-10/Week3/First-Altair-examples.html**"
    ,unsafe_allow_html=True)

st.write("Link for the font and highlight parts (5)")
st.markdown(
    "Pandas Styler - Pandas **https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Styler-Functions**"
    ,unsafe_allow_html=True)

st.write("Link for the changing to integers part (6)")
st.markdown(
    "Changing from float to int - Stack Overflow **https://stackoverflow.com/questions/21291259/convert-floats-to-ints-in-pandas**"
    ,unsafe_allow_html=True)

st.write("Link for improved Altair charts parts (7)")
st.markdown(
    "Customizing Visualizations - Altair **https://altair-viz.github.io/user_guide/customization.html**"
    ,unsafe_allow_html=True)

st.write("Link for linear regression part (8)")
st.markdown(
    "Sample code from Hands-On Machine Learning - Christopher Davis **https://christopherdavisuci.github.io/UCI-Math-10/Week5/Week5-Monday.html**"
    ,unsafe_allow_html=True)

st.write("Link for rounding part (9)")
st.markdown(
    "Rounding to 2 Decimals - Stack Overflow **https://stackoverflow.com/questions/20457038/how-to-round-to-2-decimals-with-python**"
    ,unsafe_allow_html=True)

st.write("Link for sorting dataframe (10)")
st.markdown(
    "Examples of Pandas Dataframe **https://christopherdavisuci.github.io/UCI-Math-10/Week3/Week2-Friday.html**"
    ,unsafe_allow_html=True)
