import streamlit as st
import numpy as np
import pickle
import datetime as dt
import pandas as pd
import requests
from streamlit_lottie import st_lottie

def load_resources():
  with open("model_resources.pkl", "rb") as file:
    data = pickle.load(file)
  return data

data = load_resources()

model = data['model']
OHencoder = data['one-hot']
scaler = data['scaler']

def load_lottie(url):
  r = requests.get(url)
  if r.status_code != 200:
    return None
  return r.json()

lottie_coding = load_lottie("https://assets10.lottiefiles.com/private_files/lf30_p5tali1o.json")

st.title(":house: Washington House Price Prediction")
st.markdown("<h6>Created by Bryan Tamin, Stephen Hardjadilaga, and Diven Clementius</h6>", unsafe_allow_html=True)

with st.container():
  st.write("---")
  left_column, right_column = st.columns(2)
  with left_column:
    st.header("About Us")
    st.write("##")
    st.markdown("<p style='text-align: justify;'>We are a Computer Science Student from Bina Nusantara University. For our machine learning project, we are creating a website that can help predict house prices in Washington. Since houses are a common needs in our daily life, by implementing Machine Learning Algorithm in our website, people can predict house prices based on their desired house specification ranging from Num of Bedroom, Num of Bathroom, House Location, and etc. Therefore, we hope that our website can be useful and helpful for our users. </p>", unsafe_allow_html=True)
  with right_column:
    st_lottie(lottie_coding, height=300, key="House")


st.write("---")
st.write("""### Fill in the information to Predict House Price""")

cities = (
  'Shoreline', 'Kent', 'Bellevue', 'Redmond', 'Seattle',
  'Maple Valley', 'North Bend', 'Lake Forest Park', 'Sammamish',
  'Auburn', 'Des Moines', 'Bothell', 'Federal Way', 'Kirkland',
  'Issaquah', 'Woodinville', 'Normandy Park', 'Fall City', 'Renton',
  'Carnation', 'Snoqualmie', 'Duvall', 'Burien', 'Covington',
  'Inglewood-Finn Hill', 'Kenmore', 'Newcastle', 'Black Diamond',
  'Clyde Hill', 'Algona', 'Mercer Island', 'Skykomish', 'Tukwila',
  'Vashon', 'Ravensdale', 'Yarrow Point', 'SeaTac', 'Medina',
  'Enumclaw', 'Snoqualmie Pass', 'Pacific', 'Beaux Arts Village',
  'Preston', 'Milton'
)

statezips = (
  'WA 98133', 'WA 98042', 'WA 98008', 'WA 98052', 'WA 98115',
  'WA 98038', 'WA 98045', 'WA 98155', 'WA 98105', 'WA 98074',
  'WA 98106', 'WA 98007', 'WA 98092', 'WA 98198', 'WA 98006',
  'WA 98102', 'WA 98011', 'WA 98125', 'WA 98003', 'WA 98136',
  'WA 98033', 'WA 98029', 'WA 98117', 'WA 98034', 'WA 98072',
  'WA 98023', 'WA 98107', 'WA 98166', 'WA 98116', 'WA 98024',
  'WA 98055', 'WA 98077', 'WA 98027', 'WA 98059', 'WA 98075',
  'WA 98014', 'WA 98065', 'WA 98199', 'WA 98053', 'WA 98058',
  'WA 98122', 'WA 98103', 'WA 98112', 'WA 98005', 'WA 98118',
  'WA 98177', 'WA 98004', 'WA 98019', 'WA 98144', 'WA 98168',
  'WA 98001', 'WA 98056', 'WA 98146', 'WA 98028', 'WA 98148',
  'WA 98057', 'WA 98010', 'WA 98119', 'WA 98031', 'WA 98030',
  'WA 98126', 'WA 98032', 'WA 98178', 'WA 98040', 'WA 98288',
  'WA 98108', 'WA 98109', 'WA 98070', 'WA 98051', 'WA 98188',
  'WA 98002', 'WA 98039', 'WA 98022', 'WA 98068', 'WA 98047',
  'WA 98050', 'WA 98354'
)

tof = ("No", "Yes")
with st.sidebar:
  st.write("## Additional Information:")
  city = st.selectbox("City", cities)
  statezip = st.selectbox("Statezip", statezips)
  waterfront = 1 if st.selectbox("Is it on waterfront?", tof) == "Yes" else 0
  view = 1 if st.selectbox("Is there any balcony?", tof) == "Yes" else 0
  condition = bathroom = st.select_slider("Condition",np.arange(1,6))
  yr_built = st.number_input("Year built",1800,2022,2000)
  yr_renovated = st.number_input("Year renovated",1800,2022,2000)
  listed_date = st.date_input("Listed Date")

bedroom = st.select_slider("Number of bedrooms",np.arange(0,20.1,0.5),3)
bathroom = st.select_slider("Number of bathrooms",np.arange(0,20.1,0.5),3)
sqft_lot = st.number_input("House land area (Square Feet)",0,value = 100)
sqft_above = st.number_input("House area (Square Feet)",0, value = 100)
sqft_basement = st.number_input("House basement area (Square Feet)",0,value = 100)
floor = st.select_slider("Number of floors",np.arange(0,20.1,0.5),1)




listed_day = listed_date.day
listed_month = listed_date.month

X = pd.DataFrame({
  "city":[city],
  "statezip":[statezip],
  "bedrooms":[bedroom],
  "bathrooms":[bathroom],
  "sqft_lot":[sqft_lot],
  "sqft_above":[sqft_above],
  "sqft_basement":[sqft_basement],
  "floors":[floor],
  "waterfront":[waterfront],
  "view":[view],
  "condition":[condition],
  "yr_built":[yr_built],
  "yr_renovated":[yr_renovated],
  "listed day":[listed_day],
  "listed month":[listed_month]
})

num = [
  'bedrooms',
  'bathrooms',
  'sqft_lot',
  'floors',
  'waterfront',
  'view',
  'condition',
  'sqft_above',
  'sqft_basement',
  'yr_built',
  'yr_renovated',
  'listed day',
  'listed month']

asd = pd.DataFrame(OHencoder.transform(X[['city', 'statezip']]))
X = X.reset_index(drop = True).join(asd) #One-hot
X.drop(['city', 'statezip'], axis = 1, inplace = True)

X[num] = scaler.transform(X[num]) #normalization

y = model.predict(X)

show = st.button("Show House Price Prediction")

if show:
  st.write(f"""## The house is predicted to be:""")
  st.write(f'# ${int(y):,}')