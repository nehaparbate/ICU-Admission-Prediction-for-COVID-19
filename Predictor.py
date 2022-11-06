import streamlit as st
# EDA Pkgs
import pandas as pd
import numpy as np
import webbrowser
import matplotlib

from Scripts.streamlit.proto.Image_pb2 import Image

matplotlib.use('Agg')
html_temp = """<div style="background-color:{};padding:10px;border-radius:10px">
<h1 style="color:white;text-align:center;">ICU Prediction Covid 19 </h1></div>"""

descriptive_message_temp = """<div style="background-color:silver;overflow-x: auto; 
padding:10px;border-radius:5px;margin:10px;"> <h3 style="text-align:justify;color:black;padding:10px">Definition</h3> 
<p>Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. Most people who 
fall sick with COVID-19 will experience mild to moderate symptoms and recover without special treatment.</p> </div> """

descriptive_message_temp1 = """<div style="background-color:silver;overflow-x: auto; 
padding:10px;border-radius:5px;margin:10px;">
<p>The virus that causes COVID-19 is mainly transmitted through droplets generated when an infected person coughs, 
sneezes,or exhales.</p> <p>These droplets are too heavy to hang in the air, and quickly fall on floors or surfaces. 
You can be infected by breathing in the virus if you are within close proximity of someone who has COVID-19, 
or by touching a contaminated surface and then your eyes, nose or mouth.</p> </div> """

descriptive_message_temp2 = """<div style="background-color:silver;overflow-x: auto; 
padding:10px;border-radius:5px;margin:10px;">
<p>Most common symptoms:</p>
<ul>fever</ul>
<ul>dry cough</ul>
<ul>tiredness</ul>
<p>Less common symptoms:</p>
<ul>aches and pains</ul>
<ul>sore throat</ul>
<ul>diarrhoea</ul>
<ul>conjunctivitis</ul>
<ul>headache</ul>
<ul>loss of taste or smell</ul>
<ul>a rash on skin, or discolouration of fingers or toes.<ul> </div>
 """
descriptive_message_temp4 = """<div style="background-color:blue;overflow-x: auto; 
padding:10px;border-radius:5px;margin:10px;"> <h3 style="text-align:justify;color:black;padding:10px">Post Covid Care</h3> 
<p> The coronavirus is a nasty microbe that can do a lot of damage to your body. If your infection was moderate to severe, it is possible that the virus did some amount of damage to your respiratory system.
That is why, even if your body has killed off all the viruses, you still need to shower yourself with a lot of love and care.</p>
<ul>Taking Rest</ul><ul>Having a nutritious Diet </ul><ul>Exercise a little every day</ul><ul>Plan a few memory games</ul><ul>Checking your body oxygen level</ul></div> """

descriptive_message_temp5 = """<div style="background-color:blue;overflow-x: auto; 
padding:10px;border-radius:5px;margin:10px;"> <h3 style="text-align:justify;color:black;padding:10px">PM CARES Fund</h3> 
<p>Donation - To the PEOPLE , By the People.</p>
<p>We urge all the people from all walks of life to contribute. TOGETHER we shall overcome this fight against coronavirus. </p></div>"""

maps="""
<div><iframe src="https://www.google.com/maps/embed?pb=!1m16!1m12!1m3!1d35989.848564313885!2d73.85685871811579!3d18.530401003639167!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!2m1!1sHospitals!5e0!3m2!1sen!2sin!4v1623733219179!5m2!1sen!2sin" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy"></iframe></div>
"""
prescriptive_message_temp2 ="""
<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
<ul>
<li style="text-align:justify;color:black;padding:10px">Cover coughs and sneezes with tissue,throw the tissue away.</li>
 <li style="text-align:justify;color:black;padding:10px">Wash with soap and water for at least 20 seconds.</li>
<li style="text-align:justify;color:black;padding:10px">Get Plenty of Rest</li>
<li style="text-align:justify;color:black;padding:10px">Increase Vitamin intake</li>
<li style="text-align:justify;color:black;padding:10px">Keep away from other people in the home.</li>
<li style="text-align:justify;color:black;padding:10px">Proper diet</li>
<li style="text-align:justify;color:black;padding:10px">Use separate bedding, towels and not share these.</li>
<ul>
<h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
<ul>
<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
<li style="text-align:justify;color:black;padding:10px">Take your timely medicines</li>
<ul>
</div>

"""

prescriptive_message_temp ="""
<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
<h3 style="text-align:justify;color:black;padding:10px">Medical Management</h3>
<ul>
<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
<li style="text-align:justify;color:black;padding:10px">Get Hospitalized</li>
<li style="text-align:justify;color:black;padding:10px">Monitor oxygen levels.</li>
<li style="text-align:justify;color:black;padding:10px">Maintain healthy supply of oxygen to the rest of the body.</li>
<li style="text-align:justify;color:black;padding:10px">Consider taking Plasma Therapy</li>
<ul>
</div>

"""

pune_hosp= """
<div><iframe src="https://www.google.com/maps/embed?pb=!1m16!1m12!1m3!1d60529.54604729743!2d73.85244094789813!3d18.524535061186846!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!2m1!1shospitals%20pune!5e0!3m2!1sen!2sin!4v1623740231203!5m2!1sen!2sin" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy"></iframe></div>
"""

pcmc_hosp ="""
<div><iframe src="https://www.google.com/maps/embed?pb=!1m16!1m12!1m3!1d85548.29637429863!2d73.77158321171761!3d18.630927837655!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!2m1!1shospitals%20pcmc!5e0!3m2!1sen!2sin!4v1623740650353!5m2!1sen!2sin" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy"></iframe></div>
"""

kot_hosp="""
<div><iframe src="https://www.google.com/maps/embed?pb=!1m16!1m12!1m3!1d21402.828517340015!2d73.79755204253576!3d18.505330526366418!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!2m1!1shospitals%20kothrud!5e0!3m2!1sen!2sin!4v1623740737737!5m2!1sen!2sin" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy"></iframe></div>
"""

kat_hosp="""
<div><iframe src="https://www.google.com/maps/embed?pb=!1m16!1m12!1m3!1d50918.503582805584!2d73.83319333471158!3d18.459164396419727!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!2m1!1shospitals%20katraj!5e0!3m2!1sen!2sin!4v1623740799696!5m2!1sen!2sin" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy"></iframe></div>
"""

khad_hosp="""
<div><iframe src="https://www.google.com/maps/embed?pb=!1m16!1m12!1m3!1d17991.051689682336!2d73.8417634101097!3d18.567152409930106!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!2m1!1shospitals%20khadki!5e0!3m2!1sen!2sin!4v1623740883603!5m2!1sen!2sin" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy"></iframe></div>
"""

gender_dict = {"male": 1, "female": 0}
feature_dict = {"No": 0, "Yes": 1}


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return key


def get_fvalue(val):
    feature_dict = {"No": 0, "Yes": 1}
    for key, value in feature_dict.items():
        if val == key:
            return value


def prediction1(single_sample,ch):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn import metrics as mt
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    # import a dataset
    df = pd.read_excel('C:/Users/ABCd/PycharmProjects/pythonProject/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')

    # STEP1 PREPROCESSING
    # fill forward (ffill) and backword (bfill) per patent

    data = df.groupby('PATIENT_VISIT_IDENTIFIER', as_index=False) \
        .fillna(method='ffill') \
        .fillna(method='bfill')

    # added as PATIENT_VISIT_IDENTIFIER removed during grouping

    data['PATIENT_VISIT_IDENTIFIER'] = df.PATIENT_VISIT_IDENTIFIER
    for p in data.PATIENT_VISIT_IDENTIFIER.unique():
        # get start of patent index
        start_p_index = data[data.PATIENT_VISIT_IDENTIFIER == p].index[0]
        # get last patent ICU status
        p_last_ICU = data.loc[start_p_index + 4].ICU
        # set last patent ICU status to all rows target
        data.loc[data.PATIENT_VISIT_IDENTIFIER == p, 'target'] = p_last_ICU
    data['target'] = data['target'].astype(int)
    #data[['PATIENT_VISIT_IDENTIFIER', 'WINDOW', 'ICU', 'target']][10:20]

    import pandas as pd

    # This function take a dataframe
    # as a parameter and returning list
    # of column names whose contents are identical ( duplicates) such as : .

    def get_identical_features(df):

        # Create an empty set
        duplicateColumnNames = set()

        # Iterate through all the columns
        # of dataframe
        for x in range(df.shape[1]):

            # Take column at xth index.
            col = df.iloc[:, x]

            # Iterate through all the columns in
            # DataFrame from (x + 1)th index to
            # last index
            for y in range(x + 1, df.shape[1]):

                # Take column at yth index.
                otherCol = df.iloc[:, y]

                # Check if two columns at x & y
                # index are equal or not,
                # if equal then adding
                # to the set
                if col.equals(otherCol) and df.columns.values[y] not in duplicateColumnNames:
                    duplicateColumnNames.add(df.columns.values[y])
                    # print(df.columns.values[x] ,df.columns.values[y])

        # Return list of unique column names
        # whose contents are duplicates.
        return list(duplicateColumnNames)

        # remove zero variance ( contan only one value) features

    def get_zero_variance_features(exp_data):
        remove_features = []
        for c in exp_data.columns:
            if len(data[c].value_counts()) < 2:
                remove_features.append(c)
        return remove_features

    def remove_high_corrolated_features(df, threshold=0.5):

        # get co variance of data
        coVar = df.corr()  # or df.corr().abs()
        # threshold = 0.5 #
        #1. .where(coVar != 1.0) set NaN where col and index is 1
        #2. .where(coVar >= threshold) if not greater than threshold set Nan
        #3. .fillna(0) Fill NaN with 0
        #4. .sum() convert data frame to serise with sum() and just where is co var greater than threshold sum it
        #5. > 0 convert all Series to Boolean

        coVarCols = coVar.where(coVar != 1.0).where(coVar >= threshold).fillna(0).sum() > 0
        # print(coVar.where(coVar != 1.0).where(coVar >=threshold))
        # Not Boolean Becuase we need to delete where is co var greater than threshold

        # print('High corrolated features:',list(coVarCols[coVarCols].index))

        coVarCols = ~coVarCols

        # get where you want
        columns_to_keep = list(coVarCols[coVarCols].index)
        columns_to_keep.extend(list(df.dtypes[df.dtypes == object].index))
        return df[columns_to_keep]

    def corrMatrix(df, fileName='corrMatrix', include_target=False, annot=False):
        import seaborn as sn
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 20))
        if include_target:
            columns = [x for x in df.columns if x not in ['ICU', 'PATIENT_VISIT_IDENTIFIER']]
        else:
            columns = [x for x in df.columns if x not in ['target', 'ICU', 'PATIENT_VISIT_IDENTIFIER']]

        corrMatrix = df[columns].corr()
        sn.heatmap(corrMatrix, annot=annot)
        #plt.savefig(fileName + '.png')
        return corrMatrix

    # remove identical features  and save file data & one value columns

    # remove zero variance column
    one_value_columns = get_zero_variance_features(data)
    #print('one_value_columns to remove:', len(one_value_columns))
    # print('one_value_columns:',one_value_columns)
    data.drop(columns=one_value_columns, inplace=True)

    # data includes  'WINDOW', 'ICU','PATIENT_VISIT_IDENTIFIER', 'target'
    # should subtract no. of features from 4: features without idntifiers and status

    #print(' after remove one_value_columns (-4)', data.shape)

    # remove columns have same value
    duplicateColNames = get_identical_features(data)
    #print('No. of Duplicate Columns are :', len(duplicateColNames))
    # print('Duplicate Columns are :',duplicateColNames)

    data = data.drop(columns=duplicateColNames)
    #print('Data After drop  duplicateCol are (-4):', data.shape)

    # remove columns with correlations more than 70%
    corrMatrix(data)  # requires to remove target from this digram
    data = remove_high_corrolated_features(data, 0.7)
    #print(' after remove remove_coVarCols(-4) :', data.shape)
    # print(' after remove remove_coVarCols', data.columns)
    # data.to_csv('data_remove_duplicate_columns.csv')
    # get the quantitive features from data to apply calculation on it


    qualititive_features = ['PATIENT_VISIT_IDENTIFIER', 'AGE_ABOVE65', 'GENDER',
                            'DISEASE GROUPING 1', 'DISEASE GROUPING 2', 'DISEASE GROUPING 3',
                            'DISEASE GROUPING 4', 'DISEASE GROUPING 5', 'DISEASE GROUPING 6', 'HTN',
                            'IMMUNOCOMPROMISED', 'OTHER', 'ICU', 'target',
                            'AGE_PERCENTIL', 'WINDOW']

    quantitive_features = list(set(data.columns) - set(qualititive_features))
    import statsmodels.api as sm

    # get all quantitive features  and check p-value with target (will need ICU or not )
    xx = data.loc[:, quantitive_features]
    yy = data.loc[:, 'target']

    def get_p_value(x, Y):
        regressor_OLS = sm.OLS(Y, x).fit()
        return regressor_OLS.pvalues

    p_value = get_p_value(xx.values, yy.values)
    p_value_pd = pd.Series(p_value, index=quantitive_features)
    p_value_pd = np.round(p_value_pd, 4)

    none_calc_corrMatrix = corrMatrix(data.loc[:, qualititive_features],
                                      'corrMatrix_ICU_NoN_caculated', include_target=True, annot=True)



    le = LabelEncoder()
    data['AGE_PERCENTIL'] = le.fit_transform(data['AGE_PERCENTIL'])


    data.drop(columns=['DISEASE GROUPING 2', 'DISEASE GROUPING 4', 'DISEASE GROUPING 6', 'BILLIRUBIN_MEDIAN',
                       'PATIENT_VISIT_IDENTIFIER', 'PC02_VENOUS_MEDIAN', 'P02_ARTERIAL_MEDIAN', 'WINDOW',
                       'PH_ARTERIAL_MEDIAN', 'ICU', 'AGE_PERCENTIL'], inplace=True)

    X = data.drop('target', axis='columns', inplace=False).values
    # Dependent Vector Target column values as an array
    y = data['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y)

    if ch=='RandomForest':
        from sklearn import metrics
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_jobs=64, n_estimators=200, criterion='entropy', oob_score=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        st.write("Accuracy: %.2f%%" % (acc * 100.0))
        st.write(acc)
        pred = model.predict(np.array(single_sample))

    else:
        model = XGBClassifier(learning_rate=0.1,
                          n_estimators=500,
                          max_depth=5,
                          min_child_weight=1,
                          gamma=0,
                          subsample=0.8,
                          colsample_bytree=0.8,
                          objective='binary:logistic',
                          nthread=4,
                          scale_pos_weight=1,
                          eval_metric='mlogloss'
                          )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        from sklearn.metrics import accuracy_score
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        st.write("Accuracy: %.2f%%" % (accuracy * 100.0))
        pred=model.predict(np.array(single_sample).reshape(1,-1))
    return pred


def pred_s(fl):
    from sklearn.model_selection import train_test_split
    df = pd.read_excel('C:/Users/ABCd/PycharmProjects/pythonProject/Cleaned-Data.xlsx')
    df.drop(columns=['None_Sympton', 'Severity_Mild', 'Severity_Moderate', 'Severity_None', 'Contact_Dont-Know'],
            inplace=True)
    X = df.drop('Severity_Severe', axis='columns', inplace=False).values
    # Dependent Vector Target column values as an array
    y = df['Severity_Severe'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y)

    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_jobs=64, n_estimators=200, criterion='entropy', oob_score=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    #st.write("Accuracy: %.2f%%" % (acc * 100.0))
    #st.write(acc)
    pred = model.predict(np.array(fl).reshape(1,-1))

    return pred


def hospitals():
    st.write("Maps")
    ch = st.selectbox("Select Area", ["PMC", "PCMC", "Khadki Cantonment Board" , "Kothrud" , "Katraj"])
    if ch=="PMC":
        st.markdown(pune_hosp, unsafe_allow_html=True)
    elif ch=="PCMC":
        st.markdown(pcmc_hosp, unsafe_allow_html=True)
    elif ch == "Khadki Cantonment Board":
        st.markdown(khad_hosp, unsafe_allow_html = True)
    elif ch == "Kothrud":
        st.markdown(kot_hosp, unsafe_allow_html = True)
    elif ch == "Katraj":
        st.markdown(kat_hosp, unsafe_allow_html = True)



def main():
    """ICU Prediction App"""
    st.title("ICU Prediction App")
    st.markdown(html_temp.format('royalblue'), unsafe_allow_html=True)

    menu = ["Home", "Plot", "Prediction Using Clinical Attributes", "Prediction using Symptoms" , "Helpline Centre"]

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        st.text("What is Covid?")
        st.markdown(descriptive_message_temp, unsafe_allow_html=True)
        st.text("HOW IT SPREADS?")
        st.markdown(descriptive_message_temp1, unsafe_allow_html=True)
        st.text("Symptoms")
        st.markdown(descriptive_message_temp2, unsafe_allow_html=True)



    elif choice == "Plot":
        df = pd.read_excel('C:/Users/ABCd/PycharmProjects/pythonProject/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')
        st.subheader("Data Set Used for training.")
        st.dataframe(df)
        st.subheader("Plots of some important features.")
        st.text("Gender")
        data1 = df['GENDER'].value_counts()
        st.bar_chart(data1)
        st.text("Age")
        data2 = df['AGE_PERCENTIL'].value_counts()
        st.bar_chart(data2)
        st.text("IMMUNOCOMPROMISED")
        data3 = df['IMMUNOCOMPROMISED'].value_counts()
        st.bar_chart(data3)
        st.text("AGE_ABOVE65")
        data4 = df['AGE_ABOVE65'].value_counts()
        st.bar_chart(data4)
        window_set = df[['WINDOW', 'ICU']]
        window_set = window_set[window_set['ICU'] == 1]
        data = window_set.groupby("WINDOW").sum()
        st.text("Patient Admited time window and ICU Joining")
        st.bar_chart(data)
        if st.checkbox("Other Attributes"):
            all_columns = df.columns.to_list()
            feat_choices = st.selectbox("Choose a Feature", all_columns)
            new_df = df[feat_choices].value_counts()
            st.bar_chart(new_df)

    elif choice == "Prediction Using Clinical Attributes":
        st.subheader("Prediction using clinical examinations.")

        age65 = st.radio("Is Age above 65?", tuple(feature_dict.keys()))
        sex = st.radio("Sex", tuple(gender_dict.keys()))
        grp1 = st.radio("Do You have previous Diseases(Grp1)(topographic, by bodily region or system)?", tuple(feature_dict.keys()))
        grp3 = st.radio("Do You have previous Diseases(Grp3)(physiological, by function or effect)?", tuple(feature_dict.keys()))
        grp5 = st.radio("Do You have previous Diseases(Grp5)(etiologic (causal),)?", tuple(feature_dict.keys()))
        htn = st.number_input("HTN (HyperTension)", 90.0, 140.0)
        imm = st.radio("IS immunity weak?", tuple(feature_dict.keys()))
        other = st.selectbox("OTHER Diseases?", tuple(feature_dict.keys()))
        alb = st.number_input("ALBUMIN", 34.0, 54.0)
        beve = st.number_input("BE_VENOUS", 1.0, 7.4)
        bicv = st.number_input("BIC_VENOUS Content", 1.0, 100.0)
        bla = st.number_input("BLAST Content", 1.0, 100.0)
        cal = st.number_input("CALCIUM content", 0.0, 10.3)
        cer = st.number_input("CREATININ Content", 0.0, 100.0)
        gl = st.number_input("GLUCOSE Content", 0.0, 210.0)
        inr = st.number_input("INR Content", 0.0, 3.0)
        lac = st.number_input("LACTATE Content", 4.5, 20.0)
        lin = st.number_input("LINFOCITOS_MEDIAN Content", 0.0, 6.0)
        pco = st.number_input("PC02_ARTERIAL Content", 35.0, 50.0)
        pcr = st.number_input("RT–PCR ", 10.0, 40.0)
        phv = st.number_input("PH_VENOUS Content", 1.0, 8.0)
        plat = st.number_input("PLATELETS Count", 100.0, 500.0)
        pot = st.number_input("POTASSIUM Content", 3.0, 100.0)
        sat = st.number_input("SAT02 Content", 1.0, 100.0)
        sod = st.number_input("SODIUM Content", 135.0, 296.0)
        ttp = st.number_input("TTPA_Content", 1.0, 3.0)
        ure = st.number_input("UREA Specific gravity", 0.0, 4.0)
        dim = st.number_input("DIMER Content", 0.0, 5.0)
        rep = st.number_input("RESPIRATORY_RATE",1.0, 100.0)
        oxy = st.number_input("OXYGEN_SATURATION", 1.0, 100.0)
        bp = st.number_input("BLOODPRESSURE ", 1.0, 140.0)

        feature_list = [[get_fvalue(age65), get_value(sex, gender_dict), get_fvalue(grp1), get_fvalue(grp3),
                        get_fvalue(grp5), htn, get_fvalue(imm), get_fvalue(other), alb,
                        beve, bicv, bla, cal, cer, gl, inr, lac, lin, pco, pcr, phv, plat, pot, sat, sod, ttp, ure, dim, rep, oxy, bp]]
        b=get_fvalue(age65)+get_fvalue(grp1)+get_fvalue(grp3)+get_fvalue(grp5)+get_fvalue(imm)+get_fvalue(other)
        st.write(len(feature_list))
        st.write(feature_list)


        # ML

        model_choice = st.selectbox("Select Model", ["XGBoost", "RandomForest"])
        if st.button("Predict"):
            if model_choice == "XGBoost":
                prediction = prediction1(feature_list, 'XGBoost')

            elif model_choice == "RandomForest":
                prediction = prediction1(feature_list, 'RandomForest')

            # st.write(prediction)
            # prediction_label = {"Die":1,"Live":2}
            # final_result = get_key(prediction,prediction_label)
            if prediction == 1 and b>3:
                st.warning("ICU Admission Required")

                st.subheader("Prediction using {}".format(model_choice))

                st.subheader("Prescriptive Analytics")
                st.markdown(prescriptive_message_temp, unsafe_allow_html=True)

            else:
                st.success("ICU not required")
                st.subheader("You can stay in home isolation.")
                st.markdown(prescriptive_message_temp2, unsafe_allow_html=True)
        if st.checkbox("View Hospitals"):
            hospitals()

    elif choice == "Prediction using Symptoms":
        df = pd.read_excel('C:/Users/ABCd/PycharmProjects/pythonProject/Cleaned-Data.xlsx')
        st.subheader("Data Set Used for training.")
        st.dataframe(df)
        st.subheader("Selected important features used for training.")
        df.drop(columns=['None_Sympton', 'Severity_Mild', 'Severity_Moderate', 'Severity_None','Contact_Dont-Know'],
                inplace=True)
        st.dataframe(df)
        st.subheader("Prediction using Symptoms.")
        Fever = st.radio("Do You have fever?",tuple(feature_dict.keys()))
        Tiredness = st.radio("Do You feel tiredness?",tuple(feature_dict.keys()))
        DryCough = st.radio("Do You have dry cough?",tuple(feature_dict.keys()))
        Breathing=st.radio("Do You have Difficulty in Breathing?",tuple(feature_dict.keys()))
        Sore=st.radio("Do You have Sore Throat?",tuple(feature_dict.keys()))
        Pain=st.radio("Do You have Body Ache?",tuple(feature_dict.keys()))
        Nasal=st.radio("Do You have Nasal - Congestions?",tuple(feature_dict.keys()))
        Nose=st.radio("Do You have Runny - Nose?",tuple(feature_dict.keys()))
        Diarrhea=st.radio("Do You have Diarrhea?",tuple(feature_dict.keys()))
        None_Experiencing=st.radio("Are you asymptomatic?",tuple(feature_dict.keys()))
        Age_0=st.radio("Age(0-9)?",tuple(feature_dict.keys()))
        Age_10=st.radio("Age(10-19)?",tuple(feature_dict.keys()))
        Age_20=st.radio("Age(20-24)g?",tuple(feature_dict.keys()))
        Age_25 =st.radio("Age(25-59)?",tuple(feature_dict.keys()))
        Age_60=st.radio("Age(60+)?",tuple(feature_dict.keys()))
        Gender_Female=st.radio("Female?",tuple(feature_dict.keys()))
        Gender_Male=st.radio("Male?",tuple(feature_dict.keys()))
        Gender_Transgender=st.radio("Gender_Transgende?",tuple(feature_dict.keys()))
        Contact_No=st.radio("Contact with Positive pateint?",tuple(feature_dict.keys()))
        Contact_Yes=st.radio("No contact/Don't know?",tuple(feature_dict.keys()))
        c = 0
        if get_fvalue(Fever) + get_fvalue(Fever) + get_fvalue(Tiredness) + get_fvalue(DryCough) + get_fvalue(
                Breathing) + get_fvalue(Sore) + get_fvalue(Pain) + get_fvalue(Nasal) + get_fvalue(Nose) >= 4:
            c =1
        fl = [get_fvalue(Fever),get_fvalue( Tiredness) , get_fvalue(DryCough),get_fvalue(Breathing),get_fvalue(Sore),
        get_fvalue(Pain),get_fvalue(Nasal),get_fvalue(Nose),get_fvalue(Diarrhea),get_fvalue(None_Experiencing),get_fvalue(Age_0),get_fvalue(Age_10),
        get_fvalue(Age_20),get_fvalue(Age_25),get_fvalue(Age_60),get_fvalue(Gender_Female),get_fvalue(Gender_Male),get_fvalue(Gender_Transgender),get_fvalue(Contact_No),get_fvalue(Contact_Yes)]


        if st.button("Predict"):
            p=pred_s(fl)
            if p==0 and c==0:
                st.success("ICU not required")
                st.subheader("You can stay in home isolation.")
                st.markdown(prescriptive_message_temp2, unsafe_allow_html=True)
            if c==1:
                st.warning("ICU Admission Required")
                st.subheader("Prescriptive Analytics")
                st.markdown(prescriptive_message_temp, unsafe_allow_html=True)
        if st.checkbox("View Hospitals"):
            hospitals()

    elif choice == "Helpline Centre":
        st.subheader("Information About the Availability of Beds : ")
        col2 = ['Choose from the below districts ','Pune','Thane','Navi Mumbai','Nagpur','Nashik']
        url_thane = "https://covidthane.org/availabiltyOfHospitalBeds.html"
        url_naviMum = "https://nmmchealthfacilities.com/HospitalInfo/showhospitalist"
        url_nagpur = "http://nsscdcl.org/covidbeds/AvailableHospitals.jsp"
        url_nashik = "http://covidcbrs.nmc.gov.in/home/hospitalSummary "
        url4 = "https://www.divcommpunecovid.com/ccsbeddashboard/hsr"

        l3 = st.selectbox("Select - ", col2)
        if l3 == 'Pune':
            webbrowser.open_new_tab(url4)
        elif l3 == 'Thane':
            webbrowser.open_new_tab(url_thane)
        elif l3 == 'Navi Mumbai':
            webbrowser.open_new_tab(url_naviMum)
        elif l3 == 'Nagpur':
            webbrowser.open_new_tab(url_nagpur)
        elif l3 == 'Nashik':
            webbrowser.open_new_tab(url_nashik)

        st.text(" ")
        st.text(" ")

        if st.checkbox("Contact Us : "):
            all_columns = ['Health Ministry Helpline' , 'Child Helpline','Mental Health Helpline','Senior Citizens Helpline']
            l1 = st.selectbox("Choose ", all_columns)
            if l1=="Health Ministry Helpline":
                st.text(' CALL ON : 1075')
            elif l1=="Child Helpline":
                st.text(' CALL ON : 1098')
            elif l1 == "Mental Health Helpline":
                st.text(" CALL ON : 08046110007")
            elif l1 == "Senior Citizens Helpline":
                st.text(" CALL ON : 14567")

        st.text(" ")
        st.text(" ")

        st.subheader("Vaccine Updates ")
       # st.text("Information About :")
        # st.text(" 1 )Different types of Vaccines  2 ) Vaccination Centers and Slots Available ")
        url = 'https://www.cowin.gov.in/home'
        if st.button( " Take ME to the COWIN SITE "):
            webbrowser.open_new_tab(url)
            st.text("Thank you for visiting our site .")
            st.text("It is each one of our responsibility to get vaccinated at the right time "
                    "so that")
            st.text("India will be free fron COVID - 19")

        st.text(" ")
        st.text(" ")

        cols = ['--', ' Mission Oxygen ', 'Post Covid Measures', 'Help by DONATING']
        l2 = st.selectbox("Select",cols)
        if l2 == ' Mission Oxygen ':
            url1 = 'https://www.missionoxygen.org/'
            webbrowser.open_new_tab(url1)
        elif l2 == 'Post Covid Measures' :
            url3 = 'file:///C:/Users/ABCd/OneDrive/Desktop/Fol/index42.html'
            webbrowser.open_new_tab(url3)
            st.markdown(descriptive_message_temp4, unsafe_allow_html=True)
        elif l2 == 'Help by DONATING':
            url5 = "https://www.pmcares.gov.in/en/"
            webbrowser.open_new_tab(url5)
            st.markdown(descriptive_message_temp5, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
