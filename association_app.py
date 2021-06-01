import streamlit as st
import pandas as pd
import numpy as np
import pathlib
from apyori import apriori
# from mlxtend.frequent_patterns import apriori
# from mlxtend.frequent_patterns import association_rules

st.write("""
# Association Rule 
Get rules for Product Family and Product Category based on Point-of-Sales, Year and Quarter of the corresponding year""")
st.write("---")


#project_path = pathlib.Path(__file__).parent.absolute()

df = pd.read_csv('association_rule_data.csv', index_col=0)
df['Date'] = pd.to_datetime(df['Date'])

# st.write(df.head())
# st.write(df.shape)

# ===================================================

pos_ls = sorted(list(df['Point-of-Sale_ID'].unique()))
quarter_ls = sorted(list(df['Quarter'].unique()))
year_ls = sorted(list(df['Year'].unique()))

select_pos = st.selectbox("Select POS", (pos_ls))
select_quarter = st.selectbox("Select QUARTER", (quarter_ls))
select_year = st.selectbox("Select YEAR", (year_ls))

df = df.loc[(df['Point-of-Sale_ID'] == select_pos) & (df['Quarter'] == select_quarter) & (df['Year'] == select_year)]

st.write("> You can maximize the dataframes shown by clicking on the expand button on top right")
st.write(df.head())

st.write("POS chosen:", select_pos)
st.write("Quarter chosen:", select_quarter)
st.write("Year chosen:", select_year)
st.write("Dataframe shape:",df.shape)

st.write("---")

fam_df = df.copy()

fam_df = fam_df.groupby(['Date','ProductFamily_ID'])['Sell-out units'].sum().reset_index()
fam_df.drop(columns=['Sell-out units'],inplace=True)
all_prodfam = list(fam_df['ProductFamily_ID'].unique())

# Pivot the data - lines as orders and products as columns
fam_pt = pd.pivot_table(fam_df, index='Date', columns='ProductFamily_ID', 
                    aggfunc=lambda x: 1 if len(x)>0 else 0).fillna(0)

# Apply the APRIORI algorithm to get frequent itemsets
# Rules supported in at least 5% of the transactions (more info at http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
frequent_itemsets = apriori(fam_pt, min_support=0.5, max_len = 2,use_colnames=True)

st.write("""
# Generate rules by conditions
Choose a metric and a minimum threshold to generate rules""")


# Generate the association rules - by lift
select_metric = st.selectbox("Select metric", ("lift","confidence"))
range_ls = []
for i in list(np.linspace(0,1,11)):
    i = round(i,1)
    range_ls.append(i)
select_thresh = st.selectbox("Select min. threshold", range_ls)
rulesLift = association_rules( frequent_itemsets, metric=select_metric, min_threshold=select_thresh)
rulesLift.sort_values(by=select_metric, ascending=False, inplace=True)
rulesLift.reset_index(drop=True)
rulesLift

st.write("---")
st.write("""
# Explore itemsets
Select an antecendent and a consequent and examine the itemset""")

# Add a column with the length
# frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

antecedents = st.selectbox("Select an antecedents",all_prodfam)
consequents = st.selectbox("Select a consequents",all_prodfam)

st.write("antecedent chosen",antecedents)
st.write("consequent chosen",consequents)

def get_rules(antecedents, consequents):
    var = rulesLift[(rulesLift['antecedents'] == {antecedents}) & (rulesLift['consequents'] == {consequents})]
    #st.write("=====================================")
    st.write("Support: " + str(list(var['support'])[0]))
    st.write("Rule: With " + str(antecedents) + " customer also purchase " + str(consequents))
    # second index of the inner list
    

    # third index of the list located at 0th
    # of the third index of the inner list

    st.write("Confidence: " + str(list(var['confidence'])[0]))
    st.write("Lift: " + str(list(var['lift'])[0]))
    st.write("=====================================")

try:
    get_rules(antecedents,consequents)
except IndexError:
    st.write("## There are no rules found for this specific itemset")
