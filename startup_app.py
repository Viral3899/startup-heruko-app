import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler,StandardScaler 
from sklearn.ensemble import RandomForestClassifier


st.write("""
# Startup’s Acquisition Status Prediction


The App Predict **Startup is Closed or is Running**
         
         """)

st.sidebar.header('User Input Features')


st.sidebar.markdown("""
                    **Upload csv**
                    """)
                    
uploaded_file=st.sidebar.file_uploader(("Upload your  Input csv file"), type=['csv'])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    
else:
    def user_input_features():
        category_code=st.sidebar.selectbox("Category_Code", ('advertising', 'games_video', 'other', 'software', 'mobile','network_hosting','consulting', 'enterprise', 'web', 'ecommerce','public_relations', 'biotech', 'search', 'hardware', 'education'))
        
        founded_at=st.sidebar.selectbox("founded_at",(2007, 2008, 2003, 2011, 2009, 2012, 2004, 2006, 1983, 2001, 2002,1998, 1978, 2010, 1999, 1997, 2005, 1991, 2000, 1984, 1995, 1948,1996, 1993, 1980, 1986, 1921, 1989, 1967, 1920, 1979, 1969, 1992,1994, 1987, 1957, 1954, 1976, 1901, 1982, 1988, 1975, 1933, 1990,1953, 1919, 1973, 2013, 1962, 1970, 1974, 1985, 1981, 1935, 1968,1977, 1938, 1946, 1960, 1952, 1906, 1959, 1943, 1927, 1942, 1934,1932, 1947, 1972, 1924, 1955, 1922, 1917, 1965, 1971, 1964, 1961,1945, 1903, 1910, 1907, 1908, 1963, 1916, 1939, 1915, 1958, 1923,1956, 1949, 1940, 1950, 1914, 2014, 1926, 1929, 1936, 1966, 1937,1928, 1912, 1944, 1913, 1941, 1911, 1931, 1918, 1951, 1925, 1904,1909, 1930, 1902))
        
        country_code=st.sidebar.selectbox("Country_code", ('other', 'USA', 'AUS', 'IND', 'FRA', 'ISR', 'GBR','NLD', 'DEU','CAN', 'ESP'))
        
        funding_round=st.sidebar.slider("Funding_Round",1,4,1)
        
        funding_total_usd=st.sidebar.number_input("Funding_Total_usd",300,30000000,1000000)
        
        milestone=st.sidebar.slider("Milestone of Company",1,7,1)
        
        relationships=st.sidebar.slider("Relationships of Company",1,8,1)
        
        active_days=st.sidebar.number_input("Active_days",365,43800)
        
        data={"category_code":category_code,"founded_at":founded_at,"country_code":country_code,
            "funding_rounds":funding_round,"funding_total_usd":funding_total_usd,
            "milestones":milestone,"relationships":relationships,"Active_Days":active_days
              }
        
        features=pd.DataFrame(data,index=[37740])
        return features
    
    input_df = user_input_features()
    


#Raw Data

df_company=pd.read_csv("Final_Data for EDA.csv")

company=df_company.drop('isClosed',axis=1)

df=pd.concat([company,input_df],axis=0)

st.subheader('User input features')

#encoding features


a=(pd.DataFrame(df['category_code'].value_counts()[15:].reset_index())['index'].to_list())
catcode=df['category_code'].replace(a,'other')
catcode=pd.get_dummies(catcode,prefix='category_code')

b=(pd.DataFrame(df['country_code'].value_counts()[10:].reset_index())['index'].to_list())
countrycode=df['country_code'].replace(b,'other')
countrycode=pd.get_dummies(countrycode,prefix='country_code')


df=pd.concat([df,catcode,countrycode],axis=1)
df=df.drop(['category_code','country_code'],axis=1)
    
scaler=StandardScaler()


#df[['funding_rounds','funding_total_usd','milestones','relationships','Active_Days']]=df[['funding_rounds','funding_total_usd','milestones','relationships','Active_Days']].astype('float')

#df[['founded_at','funding_rounds','funding_total_usd','milestones','relationships','Active_Days']]=scaler.fit_transform(df[['founded_at','funding_rounds','funding_total_usd','milestones','relationships','Active_Days']])


    
df=df.loc[37740:]


st.write("""
         Data After Encoding
         
         """)



if uploaded_file is not None:
    st.write(df)
    
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

load_model=pickle.load(open("Startup's Acquisition Status.pkl",'rb'))


prediction=load_model.predict(df)

prediction_proba=load_model.predict_proba(df)


st.subheader('Prdiction')
status=np.array(['Closed','Running'])
st.write(status[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
            
