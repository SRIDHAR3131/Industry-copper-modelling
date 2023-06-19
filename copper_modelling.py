import streamlit as st
from streamlit_option_menu import option_menu
from animation import*
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler,LabelEncoder

from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

#page congiguration
st.set_page_config(page_title= "Copper Modelling",
                   page_icon= 'random',
                   layout= "wide",)


#=========hide the streamlit main and footer
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

left,right=st.columns([1,3])

with left:
    url_link0="https://assets5.lottiefiles.com/packages/lf20_alg1vyxd.json"
    st_lottie = lottie_home1(url_link0)

with right:
    st.markdown("<h1 style='text-align: center; color: red;'>COPPER SELLING PRICE PREDICTION AND STATUS</h1>",
                unsafe_allow_html=True)

    selected = option_menu(None, ['HOME',"PRICE PREDICTION","STATUS",],
                           icons=["house",'cash-coin','trophy'],orientation='horizontal',default_index=0)

    if selected=='HOME':
        with left:
            url_link1 = "https://assets1.lottiefiles.com/private_files/lf30_kit9njnq.json"
            st_lottie = lottie_home1(url_link1)

        st.write('## **WELCOME TO INDUSTRIAL COPPER MODELLING**')
        st.markdown('##### ***This project focuses on modelling industrial copper data using Python and various libraries such as pandas, numpy, scikit-learn. The objective of the project is to preprocess the data, handle missing values, detect outliers, and handle skewness. Additionally, regression and classification models will be built to predict the selling price and determine if a sale was won or lost. The trained models will be saved as a pickle file for later use in a Streamlit application.***')
        left ,right=st.columns([2,2])
        with left:
            st.write('### TECHNOLOGY USED')
            st.write('- PYTHON   (PANDAS, NUMPY)')
            st.write('- SCIKIT-LEARN')
            st.write('- DATA PREPROCESSING')
            st.write('- EXPLORATORY DATA ANALYSIS')
            st.write('- STREAMLIT')

        with right:
            st.write("### MACHINE LEARNING MODEL")
            st.write('#### REGRESSION - ***:red[EXTRATREEREGRESSOR]***')
            st.write('- The ExtraTree Regressor is an ensemble learning method that belongs to the tree-based family of models.')
            st.write('#### CLASSIFICATION - ***:green[RANDOMFORESTCLASSIFIER]***')
            st.write('- The RandomForestClassifier is an ensemble learning method that combines multiple decision trees to create a robust and accurate classification model.')




    if selected=='PRICE PREDICTION':
        with left:
            url_link2 = "https://assets8.lottiefiles.com/packages/lf20_OPFirj1e4d.json"
            st_lottie = lottie_price1(url_link2)

        item_list=['W', 'S', 'Others', 'PL', 'WI', 'IPL']
        status_list=['Won', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised','Offered', 'Offerable']
        country_list=['28', '32', '38', '78', '27', '30', '25', '77', '39', '40', '26', '84', '80', '79','113', '89']
        application_list=[10, 41, 28, 59, 15, 4, 38, 56, 42, 26, 27, 19, 20, 66,
                          29, 22, 40, 25, 67, 79, 3, 99, 2, 5,39, 69, 70, 65, 58, 68]

        product_list=[1670798778, 1668701718, 628377, 640665, 611993, 1668701376,
                      164141591, 1671863738, 1332077137,     640405, 1693867550, 1665572374,
                      1282007633, 1668701698, 628117, 1690738206, 628112, 640400,
                      1671876026, 164336407, 164337175, 1668701725, 1665572032, 611728,
                      1721130331, 1693867563, 611733, 1690738219, 1722207579, 929423819,
                      1665584320, 1665584662, 1665584642]
        st.write(
            '##### ***<span style="color:yellow">Fill all the fields and Press the below button to view the :red[predicted price]   of copper</span>***',
            unsafe_allow_html=True)

        c1,c2,c3=st.columns([2,2,2])
        with c1:
            quantity=st.text_input('Enter Quantity  (Min:611728 & Max:1722207579) in tons')
            thickness = st.text_input('Enter Thickness (Min:0.18 & Max:400)')
            width = st.text_input('Enter Width  (Min:1, Max:2990)')


        with c2:
            country = st.selectbox('Country Code', country_list)
            status = st.selectbox('Status', status_list)
            item = st.selectbox('Item Type', item_list)

        with c3:
            application = st.selectbox('Application Type', application_list)
            product = st.selectbox('Product Reference', product_list)
            item_order_date = st.date_input("Order Date", datetime.date(2023, 1, 1))
            item_delivery_date = st.date_input("Estimated Delivery Date", datetime.date(2023, 1, 1))
        with c1:
            st.write('')
            st.write('')
            st.write('')
            if st.button('PREDICT PRICE'):
                data = []
                with open('country.pkl', 'rb') as file:
                    encode_country = pickle.load(file)
                with open('status.pkl', 'rb') as file:
                    encode_status = pickle.load(file)
                with open('item type.pkl', 'rb') as file:
                    encode_item = pickle.load(file)
                with open('scaling.pkl', 'rb') as file:
                    scaled_data = pickle.load(file)
                with open('Extratreeregressor.pkl', 'rb') as file:
                    trained_model = pickle.load(file)

                transformed_country = encode_country.transform(country_list)
                encoded_ct = None
                for i, j in zip(country_list, transformed_country):
                    if country == i:
                        encoded_ct = j
                        break
                else:
                    st.error("Country not found.")
                    exit()

                transformed_status = encode_status.transform(status_list)
                encode_st = None
                for i, j in zip(status_list, transformed_status):
                    if status == i:
                        encode_st = j
                        break
                else:
                    st.error("Status not found.")
                    exit()


                transformed_item = encode_item.transform(item_list)
                encode_it = None
                for i, j in zip(item_list, transformed_item):
                    if item == i:
                        encode_it = j
                        break
                else:
                    st.error("Item type not found.")
                    exit()



                order = datetime.datetime.strptime(str(item_order_date), "%Y-%m-%d")
                delivery = datetime.datetime.strptime(str(item_delivery_date), "%Y-%m-%d")
                day = delivery - order


                data.append(quantity)
                data.append(thickness)
                data.append(width)
                data.append(encoded_ct)
                data.append(encode_st)
                data.append(encode_it)
                data.append(application)
                data.append(product)
                data.append(day.days)

                x = np.array(data).reshape(1, -1)
                pred_model= scaled_data.transform(x)
                price_predict= trained_model.predict(pred_model)
                predicted_price = str(price_predict)[1:-1]
                st.write(f'Predicted Selling Price : :green[₹] :green[{predicted_price}]')

        st.info("The Predicted selling price may be differ from various reason like Supply and Demand Imbalances,Infrastructure and Transportation etc..",icon='ℹ️')



    if selected=='STATUS':
        with left:
             #url_link3 = "https://assets9.lottiefiles.com/packages/lf20_lw4olqnf.json"
             url_link3='https://assets8.lottiefiles.com/private_files/lf30_vsr6pvvl.json'
             lottie_status1(url_link3)
             url_link4= "https://assets1.lottiefiles.com/private_files/lf30_by9lgy8q.json"
             lottie_status1(url_link4)

        item_list_cls = ['W', 'S', 'Others', 'PL', 'WI', 'IPL']
        country_list_cls = ['28', '32', '38', '78', '27', '30', '25', '77', '39', '40', '26', '84', '80', '79', '113', '89']
        application_list_cls = [10, 41, 28, 59, 15, 4, 38, 56, 42, 26, 27, 19, 20, 66,
                            29, 22, 40, 25, 67, 79, 3, 99, 2, 5, 39, 69, 70, 65, 58, 68]
        product_list_cls = [1670798778, 1668701718, 628377, 640665, 611993, 1668701376,
                            164141591, 1671863738, 1332077137, 640405, 1693867550, 1665572374,
                            1282007633, 1668701698, 628117, 1690738206, 628112, 640400,
                            1671876026, 164336407, 164337175, 1668701725, 1665572032, 611728,
                            1721130331, 1693867563, 611733, 1690738219, 1722207579, 929423819,
                            1665584320, 1665584662, 1665584642]

        st.write(
                '##### ***<span style="color:yellow">Fill all the fields and Press the below button to view the status :red[WON / LOST] of copper in the desired time range</span>***',
                unsafe_allow_html=True)

        cc1, cc2, cc3 = st.columns([2, 2, 2])
        with cc1:
            quantity_cls = st.text_input('Enter Quantity  (Min:611728 & Max:1722207579) in tons')
            thickness_cls = st.text_input('Enter Thickness (Min:0.18 & Max:400)')
            width_cls= st.text_input('Enter Width  (Min:1, Max:2990)')

        with cc2:
            selling_price_cls= st.text_input('Enter Selling Price  (Min:1, Max:100001015)')
            item_cls = st.selectbox('Item Type', item_list_cls)
            country_cls= st.selectbox('Country Code', country_list_cls)

        with cc3:
            application_cls = st.selectbox('Application Type', application_list_cls)
            product_cls = st.selectbox('Product Reference', product_list_cls)
            item_order_date_cls = st.date_input("Order Date", datetime.date(2023, 1, 1))
            item_delivery_date_cls = st.date_input("Estimated Delivery Date", datetime.date(2023,1, 1))
        with cc1:
            st.write('')
            st.write('')
            st.write('')
            if st.button('PREDICT STATUS'):
                data_cls = []
                with open('country.pkl', 'rb') as file:
                    encode_country_cls = pickle.load(file)
                with open('item type.pkl', 'rb') as file:
                    encode_item_cls = pickle.load(file)
                with open('scaling_classify.pkl', 'rb') as file:
                    scaled_data_cls = pickle.load(file)
                with open('randomforest_classification.pkl', 'rb') as file:
                    trained_model_cls = pickle.load(file)

                transformed_country_cls = encode_country_cls.transform(country_list_cls)
                encoded_ct_cls = None
                for i, j in zip(country_list_cls, transformed_country_cls):
                    if country_cls == i:
                        encoded_ct_cls = j
                        break
                else:
                    st.error("Country not found.")
                    exit()

                transformed_item_cls = encode_item_cls.transform(item_list_cls)
                encode_it_cls = None
                for i, j in zip(item_list_cls, transformed_item_cls):
                    if item_cls == i:
                        encode_it_cls = j
                        break
                else:
                    st.error("Item type not found.")
                    exit()

                order_cls = datetime.datetime.strptime(str(item_order_date_cls), "%Y-%m-%d")
                delivery_cls = datetime.datetime.strptime(str(item_delivery_date_cls), "%Y-%m-%d")
                day_cls = delivery_cls- order_cls

                data_cls.append(quantity_cls)
                data_cls.append(thickness_cls)
                data_cls.append(width_cls)
                data_cls.append(selling_price_cls)
                data_cls.append(encoded_ct_cls)
                data_cls.append(encode_it_cls)
                data_cls.append(application_cls)
                data_cls.append(product_cls)
                data_cls.append(day_cls.days)

                x_cls = np.array(data_cls).reshape(1, -1)
                scaling_model_cls = scaled_data_cls.transform(x_cls)
                pred_status = trained_model_cls.predict(scaling_model_cls)
                if pred_status==6:
                    st.write(f'Predicted Status : :green[WON]')
                else:
                    st.write(f'Predicted Status : :red[LOST]')

        st.info("The Predicted Status may be differ from various reason like Supply and Demand Imbalances,Infrastructure and Transportation etc..",icon='ℹ️')





