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

st.write("Reading the district level csv file and cleaning it from not applicable values")

df_district = pd.read_csv(r"C:\Users\alexa\Downloads\districts.csv")
df_district = df_district[df_district.notna().all(axis=1)].copy()

st.write("Checking the shape of the dataframe")

df_district.shape

st.write("Please enter your state choice with capital letters only on the first letters of the words")

state_choice = st.text_input("Choose the state")

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

st.write("Begin clustering data based on their revenues and expenditures statistics")

st.write("To gather the revenues and expenditures statistics, the numeric columns are extracted.")

numeric_cols_dist = [c for c in df_dchoice.columns if is_numeric_dtype(df_dchoice[c])]

num_df_dchoice = df_dchoice[numeric_cols_dist].copy()

num_df_dchoice = num_df_dchoice[num_df_dchoice["PROFIT/LOSS"].notna()].copy()

num_df_dchoice

st.write("To begin clustering, the application uses Kmeans to cluster data whose distances are the nearest to some different assigned groups")

num_of_clusters = st.slider("Please choose the number of clusters",0,10)

kmeans = KMeans(num_of_clusters)

kmeans.fit(num_df_dchoice)

df_dchoice["CLUSTER"] = kmeans.predict(num_df_dchoice)

st.write("Here is the chart of the clusters.")

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

st.write("Reading the state level csv file and cleaning it from not applicable values")

df_state = pd.read_csv(r"C:\Users\alexa\Downloads\states.csv")
df_state = df_state[df_state.notna().all(axis=1)].copy()

df_state.shape

df_schoice = df_state[df_state["STATE"] == state_choice].copy()

df_schoice["PROFIT/LOSS"] = df_schoice.loc[:,"TOTAL_REVENUE"] - df_schoice.loc[:,"TOTAL_EXPENDITURE"]

sorted_df_schoice = df_schoice.sort_values("YEAR",ascending=True)

st.write("Here is the dataframe of the state level dataframe sorted ascendingly based on year")

sorted_df_schoice

numeric_cols_stat = [c for c in sorted_df_schoice.columns if is_numeric_dtype(sorted_df_schoice[c])]

num_df_schoice = sorted_df_schoice[numeric_cols_stat].copy()

num_df_schoice = num_df_schoice.astype(int)

def style_negative(value, props=''):
    return props if value < 0 else None

styled_df_schoice = num_df_schoice.style.applymap(style_negative, props='color:purple;')

styled_df_schoice

def highlight_max(value, props=''):
    return np.where(value == np.nanmax(value.values), props, '')

def highlight_min(value, props=''):
    return np.where(value == np.nanmin(value.values), props, '')

styled2_df_schoice = styled_df_schoice.apply(highlight_max, props='color:white;background-color:green', axis=0)
styled2_df_schoice = styled_df_schoice.apply(highlight_min, props='color:white;background-color:red', axis=0)

styled2_df_schoice

state_chart = alt.Chart(df_schoice).mark_circle().encode(
    x = alt.X("YEAR",scale=alt.Scale(zero=False)),
    y = "PROFIT/LOSS",
    color = alt.Color("TOTAL_REVENUE",scale=alt.Scale(scheme='accent')),
    size = alt.Size("ENROLL",scale=alt.Scale(zero=False)),
    tooltip = ["YEAR","ENROLL","TOTAL_REVENUE","TOTAL_EXPENDITURE","PROFIT/LOSS"],
).properties(
    width = 500,
    height = 500,
    title = "State Level Educational Finances"
)

st.altair_chart(state_chart)

X = np.array(num_df_schoice["YEAR"]).reshape(-1,1)
y = np.array(num_df_schoice["PROFIT/LOSS"]).reshape(-1,1)

model = sklearn.linear_model.LinearRegression()

model.fit(X, y)

model.coef_

model.intercept_

year_pred = st.text_input("Choose the year you want to predict the profit or loss")

X_new = [[year_pred]]

profit_loss_pred = round(float(model.predict(X_new)),2)

print(f"For the year {year_pred}, the predicted profit or loss for the state {state_choice} is {profit_loss_pred} dollars.")
    
st.write("References")

st.markdown(
    "K-Means clustering Using scikit-learn - Christopher Davis **https://christopherdavisuci.github.io/UCI-Math-10/Week7/Week6-Friday.html**"
    ,unsafe_allow_html=True)