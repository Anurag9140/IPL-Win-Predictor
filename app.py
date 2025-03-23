import pandas as pd
import numpy as np

match=pd.read_csv("matches.csv")
delivery=pd.read_csv("Data/deliveries.csv")

match.head()

total_score_data=delivery.groupby(['match_id','inning']).sum()["total_runs"].reset_index()
total_score_data.head()

total_score_data=total_score_data[total_score_data['inning']==1]
total_score_data.head()

match_df=match.merge(total_score_data[['match_id','total_runs']],left_on='id',right_on='match_id')

match_df['team1'].unique()

teams=[
    'Royal Challengers Bangalore',
    'Mumbai Indians',
    'Chennai Super Kings',
    'Kolkata Knight Riders',
    'Rajasthan Royals',
    'Gujarat Titans',
    'Lucknow Super Giants',
    'Punjab Kings',
    'Sunrisers Hyderabad',
    'Delhi Capitals',
]

match_df['team1']=match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2']=match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1']=match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2']=match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

match_df['team1']=match_df['team1'].str.replace('Kings XI Punjab','Punjab Kings')
match_df['team2']=match_df['team2'].str.replace('Kings XI Punjab','Punjab Kings')

match_df=match_df[match_df['team1'].isin(teams)]
match_df=match_df[match_df['team2'].isin(teams)]



match_df['method']=match_df['method'].fillna(0)
match_df.head()

match_df=match_df[match_df['method']==0]
match_df['method'].value_counts()
match_df=match_df[['city','total_runs','match_id','winner']]

delivery_df=match_df.merge(delivery,on='match_id')

delivery_df=delivery_df[delivery_df['inning']==2]

delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()

delivery_df['runs_left']=delivery_df['total_runs_x']-delivery_df['current_score']

delivery_df['balls_left']=120 - (delivery_df['over']*6 + delivery_df['ball'])

delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x if x == "0" else "1")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum().values
delivery_df['wickets'] = 10 - wickets
delivery_df.head()

delivery_df['crr']=(delivery_df['current_score']*6)/(120-delivery_df['balls_left'])
delivery_df['required_rr']=(delivery_df['runs_left']*6)/delivery_df['balls_left']

def result(row):
    return 1 if row['batting_team']==row['winner'] else 0

delivery_df['result']=delivery_df.apply(result,axis=1)

final_df=delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','required_rr','result']]

final_df=final_df.sample(final_df.shape[0])
final_df.sample()
final_df.dropna(inplace=True)
final_df=final_df[final_df['balls_left']!=0]

X=final_df.iloc[:,:-1]
y=final_df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
transformer=ColumnTransformer(
    [
        ('transformer',OneHotEncoder(sparse_output=False,drop='first'),['batting_team','bowling_team','city'])
    ]
,remainder='passthrough')

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe=Pipeline(steps=[
    ('step1',transformer),
    ('step2',LogisticRegression(solver='liblinear'))
])

pipe.fit(X_train,y_train)

y_pred=pipe.predict(X_test)
from sklearn.metrics import accuracy_score,r2_score
accuracy_score(y_test,y_pred)



import streamlit as st
import pandas as pd

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('Training/pipe.pkl','rb'))
st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target')

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'required_rr':[rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")


