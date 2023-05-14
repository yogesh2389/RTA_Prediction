# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
#import pandas as pd
import numpy as np 
import datetime
import joblib
from prediction import get_prediction, ordinal_encoder

# Creating options list for dropdown menu

day_of_week = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
age_band_of_driver = ['18-30', '31-50', 'Over 51', 'Under 18', 'Unknown']
sex_of_driver = ['Male', 'Female', 'Unknown']
educational_level = ['Above high school', 'Junior high school', 'Elementary school', 'High school',
                     'Unknown', 'Illiterate', 'Writing & reading'] 
Vehicle_driver_relation = ['Employee' 'Unknown' 'Owner' 'Other']
driving_experience = ['Below 1yr','1-2yr', '2-5yr', '5-10yr', 'Above 10yr', 'No Licence', 'unknown'] 
type_of_vehicle = ['Automobile', 'Public (> 45 seats)', 'Lorry (41?100Q)', 'Public (13?45 seats)',
                   'Lorry (11?40Q)', 'Long lorry', 'Public (12 seats)', 'Taxi', 'Pick up upto 10Q',
                   'Stationwagen', 'Ridden horse', 'Other', 'Bajaj', 'Turbo', 'Motorcycle',
                   'Special vehicle', 'Bicycle']
Owner_of_vehicle = ['Owner' 'Governmental' 'Organization' 'Other'] 
Service_year_of_vehicle = ['Above 10yr' '5-10yrs' '1-2yr' '2-5yrs' 'Unknown' 'Below 1yr']
Defect_of_vehicle = ['No defect' '7' '5'] 
area_accident_occured = ['Residential areas', 'Office areas', '  Recreational areas',
                         ' Industrial areas', 'Other', ' Church areas', '  Market areas', 'Unknown',
                         'Rural village areas', ' Outside rural areas', ' Hospital areas',
                         'School areas', 'Rural village areasOffice areas', 'Recreational areas'] 
lanes_or_Medians = ['Undivided Two way', 'other', 'Double carriageway (median)', 'One way',
                    'Two-way (divided with solid lines road marking)',
                    'Two-way (divided with broken lines road marking)', 'Unknown']
road_allignment = ['Tangent road with flat terrain', 'Tangent road with mild grade and flat terrain',
                   'Escarpments', 'Tangent road with rolling terrain', 'Gentle horizontal curve',
                    'Tangent road with mountainous terrain and',
                    'Steep grade downward with mountainous terrain', 'Sharp reverse curve',
                    'Steep grade upward with mountainous terrain'] 
types_of_Junction = ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'Unknown', 'T Shape',
                     'X Shape'] 
road_surface_type = ['Asphalt roads', 'Earth roads', 'Asphalt roads with some distress',
                     'Gravel roads', 'Other'] 
Road_surface_conditions = ['Dry' 'Wet or damp' 'Snow' 'Flood over 3cm. deep'] 
Light_conditions = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
                    'Darkness - lights unlit'] 
weather_conditions = ['Normal', 'Raining', 'Raining and Windy', 'Cloudy', 'Other', 'Windy', 'Snow',
                      'Unknown', 'Fog or mist'] 
type_of_collision = ['Collision with roadside-parked vehicles', 'Vehicle with vehicle collision',
                     'Collision with roadside objects', 'Collision with animals', 'Other', 'Rollover',
                     'Fall from vehicles', 'Collision with pedestrians', 'With Train', 'Unknown'] 
Number_of_vehicles_involved = [2, 1, 3, 6, 4, 7, 5] 
Number_of_casualties = [2, 1, 3, 4, 6, 5, 8, 7] 
vehicle_movement = ['Going straight', 'U-Turn', 'Moving Backward', 'Turnover', 'Waiting to go',
                    'Getting off', 'Reversing', 'Unknown', 'Parked', 'Stopping', 'Overtaking', 'Other',
                    'Entering a junction'] 
Casualty_class = ['na' 'Driver or rider' 'Pedestrian' 'Passenger'] 
Sex_of_casualty = ['na' 'Male' 'Female'] 
age_band_of_casualty = ['na', '31-50', '18-30', 'Under 18', 'Over 51', '5'] 
Casualty_severity = ['na' '3' '2' '1'] 
work_of_casuality = ['Driver', 'Other', 'Unemployed', 'Employee', 'Self-employed', 'Student', 'Unknown'] 
fitness_of_casuality = ['Normal', 'Deaf', 'Other', 'Blind', 'NormalNormal'] 
pedestrian_movement = ['Not a Pedestrian', "Crossing from driver's nearside",
                       'Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle',
                       'Unknown or other',
                       'Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle',
                       'In carriageway,statioNot a Pedestrianry - not crossing  (standing or playing)',
                       'Walking along in carriageway, back to traffic',
                       'Walking along in carriageway, facing traffic',
                       'In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle'] 
cause_of_accident = ['Moving Backward', 'Overtaking', 'Changing lane to the left', 'Changing lane to the right',
                     'Overloading', 'Other', 'No priority to vehicle', 'No priority to pedestrian',
                     'No distancing', 'Getting off the vehicle improperly', 'Improper parking', 'Overspeed',
                     'Driving carelessly', 'Driving at high speed', 'Driving to the left', 'Unknown',
                     'Overturning', 'Turnover', 'Driving under the influence of drugs' 'Drunk driving'] 
Accident_severity = ['Slight Injury' 'Serious Injury' 'Fatal injury'] 

features = ['Day_of_week', 'Age_band_of_driver', 'Sex_of_driver', 'Educational_level',
            'Vehicle_driver_relation', 'Driving_experience', 'Type_of_vehicle', 'Owner_of_vehicle',
            'Service_year_of_vehicle', 'Defect_of_vehicle', 'Area_accident_occured', 
            'Lanes_or_Medians', 'Road_allignment', 'Types_of_Junction', 'Road_surface_type',
            'Road_surface_conditions', 'Light_conditions', 'Weather_conditions', 'Type_of_collision',
            'Number_of_vehicles_involved', 'Number_of_casualties', 'Vehicle_movement', 'Casualty_class',
            'Sex_of_casualty', 'Age_band_of_casualty', 'Casualty_severity', 'Work_of_casuality', 
            'Fitness_of_casuality', 'Pedestrian_movement', 'Cause_of_accident', 'Accident_severity']

def main():
    st.title("Accident Severity Predictor 🚦 🚧 ")
    st.image(r"Dataset/RTA_image.jpg", width = 650)
    #with st.form('Prediction_form'):
    st.subheader('Enter the input for the following features: ')
    #Time = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
    Time = st.time_input('Incident Time:', datetime.time(00, 00))
    Day_of_week = st.selectbox("Select Day of the Week: ", options = day_of_week)
    Age_band_of_driver = st.selectbox("Select Driver Age: ", options = age_band_of_driver)
    Sex_of_driver = st.radio('Gender of Driver:', options = sex_of_driver)
    Educational_level = st.selectbox("Select Driver Age: ", options = educational_level)
    #    Educational_level = st.selectbox("Education of Driver: ", options = educational_level)
    Vehicle_driver_relation = st.radio('Vehicle driver relation:', ['Employee', 'Unknown',
                                                                        'Owner', 'Other'])
    Driving_experience = st.selectbox("Exp. of Driver:", options = driving_experience)
    Type_of_vehicle = st.selectbox("Type of vehicle:", options = type_of_vehicle)
    Owner_of_vehicle = st.radio('Vehicle Proprietary Rights:', ['Owner', 'Governmental',
                                                                    'Organization', 'Other'] )
    Service_year_of_vehicle = st.radio('Service year of vehicle:', ['Above 10yr', '5-10yrs',
                                                                        '1-2yr','2-5yrs','Unknown','Below 1yr'])
    Defect_of_vehicle = st.slider("Vehicle Defect: ", 1, 7, value=0, format="%d")
    Area_accident_occured = st.selectbox("Location of Accident:", options = area_accident_occured)
    Lanes_or_Medians = st.selectbox("Lanes type:", options = lanes_or_Medians)
    Road_allignment = st.selectbox("Road type:", options = road_allignment)
    Types_of_Junction = st.selectbox("Junction type:", options = types_of_Junction)
    Road_surface_type = st.selectbox("Surface of road:", options = road_surface_type)
    Road_surface_conditions = st.radio('Road conditions:', ['Dry', 'Wet or damp', 'Snow',
                                                                        'Flood over 3cm. deep'])
    Light_conditions = st.radio('Light conditions:', ['Daylight', 'Darkness - lights lit',
                                                'Darkness - no lighting', 'Darkness - lights unlit'] )
    Weather_conditions = st.selectbox("Weather conditions:", options = weather_conditions)
    Type_of_collision = st.selectbox("Collision type:", options = type_of_collision)
    Number_of_vehicles_involved = st.slider("Vehicle involvement: ", 1, 7, value=0, format="%d")
    Number_of_casualties = st.slider("Casualties count: ", 1, 8, value=0, format="%d")
    Vehicle_movement = st.selectbox("Vehicle movement:", options = vehicle_movement)
    Casualty_class = st.radio('Casualty class:', ['na', 'Driver or rider', 'Pedestrian', 'Passenger']) 
    Sex_of_casualty = st.radio('Casualty gender:', ['na', 'Male', 'Female'])
    Age_band_of_casualty = st.selectbox("Casualty age band:", options = age_band_of_casualty)
    Casualty_severity = st.radio('Casualty severity:', ['na', '3', '2', '1'] )
    Work_of_casuality = st.selectbox("Work of casuality:", options = work_of_casuality)
    Fitness_of_casuality = st.selectbox("Fitness of casuality:", options = fitness_of_casuality)
    Pedestrian_movement = st.selectbox("Pedestrian movement:", options = pedestrian_movement)
    Cause_of_accident = st.selectbox("Cause of accident:", options = cause_of_accident)
        
    agreed = st.checkbox('I agree to the terms and conditions for obtaining the predictions.')
        
    if agreed:
        submit = st.button("Predict")
        if submit:
            Day_of_week = ordinal_encoder(Day_of_week, day_of_week)
            Age_band_of_driver = ordinal_encoder(Age_band_of_driver, age_band_of_driver)
            Sex_of_driver = ordinal_encoder(Sex_of_driver, sex_of_driver)
            Educational_level = ordinal_encoder(Educational_level, educational_level)
            Vehicle_driver_relation = ordinal_encoder(Vehicle_driver_relation, )
            Driving_experience = ordinal_encoder(Driving_experience, driving_experience)
            Type_of_vehicle = ordinal_encoder(Type_of_vehicle, type_of_vehicle)
            Owner_of_vehicle = ordinal_encoder(Owner_of_vehicle, )
            Service_year_of_vehicle = ordinal_encoder(Service_year_of_vehicle, )
            Defect_of_vehicle = ordinal_encoder(Defect_of_vehicle, )
            Area_accident_occured = ordinal_encoder(Area_accident_occured, area_accident_occured)
            Lanes_or_Medians = ordinal_encoder(Lanes_or_Medians, lanes_or_Medians)
            Road_allignment = ordinal_encoder(Road_allignment, )
            Types_of_Junction = ordinal_encoder(Types_of_Junction, )
            Road_surface_type = ordinal_encoder(Road_surface_type, )
            Road_surface_conditions = ordinal_encoder(Road_surface_conditions, )
            Light_conditions = ordinal_encoder(Light_conditions, )
            Weather_conditions = ordinal_encoder(Weather_conditions, )
            Type_of_collision = ordinal_encoder(Type_of_collision, )
            Number_of_vehicles_involved = ordinal_encoder(Number_of_vehicles_involved, )
            Number_of_casualties = ordinal_encoder(Number_of_casualties, )
            Vehicle_movement = ordinal_encoder(Vehicle_movement, )
            Casualty_class = ordinal_encoder(Casualty_class, )
            Sex_of_casualty = ordinal_encoder(Sex_of_casualty, )
            Age_band_of_casualty = ordinal_encoder(Age_band_of_casualty, )
            Casualty_severity = ordinal_encoder(Casualty_severity, )
            Work_of_casuality = ordinal_encoder(Work_of_casuality, )
            Fitness_of_casuality = ordinal_encoder(Fitness_of_casuality, )
            Pedestrian_movement = ordinal_encoder(Pedestrian_movement, )
            Cause_of_accident = ordinal_encoder(Cause_of_accident, )
     
        
                  
            data = np.array(['Day_of_week', 'Age_band_of_driver', 'Sex_of_driver',
                'Educational_level', 'Vehicle_driver_relation', 'Driving_experience',
                'Type_of_vehicle', 'Owner_of_vehicle', 'Service_year_of_vehicle',
                'Defect_of_vehicle', 'Area_accident_occured', 'Lanes_or_Medians',
                'Road_allignment', 'Types_of_Junction', 'Road_surface_type',
                'Road_surface_conditions', 'Light_conditions', 'Weather_conditions',
                'Type_of_collision', 'Number_of_vehicles_involved',
                'Number_of_casualties', 'Vehicle_movement', 'Casualty_class',
                'Sex_of_casualty', 'Age_band_of_casualty', 'Casualty_severity',
                'Work_of_casuality', 'Fitness_of_casuality', 'Pedestrian_movement',
                'Cause_of_accident']).reshape(1,-1)
    
        
            model = joblib.load(r'ExtraTreesClassifier_model.joblib')
        
            pred = get_prediction(data=data, model=model)
        
            st.write(f"The predicted severity is:  {pred[0]}")
            #st.write('Prediction enabled!')
    else:
        st.warning("Prediction disabled.You must agree to the terms and conditions to use this app.")
        #st.write('Prediction disabled.')
    
        #submit = st.form_submit_button("Predict")
        #else:
            #st.warning("You must agree to the terms and conditions to use this app.")
    
    


if __name__ == "__main__":
    main()


