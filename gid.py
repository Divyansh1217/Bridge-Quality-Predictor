import pandas as pd
data=pd.read_csv("bridge_data.csv")
print(data.head())
print(data.info())
data["Material"]=data["Material"].map({"Steel":0,"Concrete":1,"Wood":2})
data["Weather_Conditions"]=data["Weather_Conditions"].map({"Sunny":0,"Windy":1,"Rainy":2,"Cloudy":3,"Snowy":4})
data["Construction_Quality"]=data["Construction_Quality"].map({"Bad":0,"Good":1})
data["Bridge_Design"]=data["Bridge_Design"].map({"Arch":0,"Beam":1,"Truss":2})
data["Material_Composition"]=data["Material_Composition"].map({"Steel 70%, Concrete 30%":0,"Concrete 80%, Wood 20%":1})
data["Collapse_Status"]=data["Collapse_Status"].map({"Standing":0,"Collapsed":1})

x=data.drop(["Bridge_ID","Location","Collapse_Status","Maintenance_History","Material_Composition"],axis=1)


y=data[["Collapse_Status"]]
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

xtrain,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()
from sklearn.metrics import accuracy_score,jaccard_score
model.fit(xtrain,y_train)
pre=model.predict(x_test)
sco=accuracy_score(y_test,pre)
print(sco)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(xtrain,y_train)
pre=model.predict(x_test)
sco=jaccard_score(y_test,pre)
print(sco)