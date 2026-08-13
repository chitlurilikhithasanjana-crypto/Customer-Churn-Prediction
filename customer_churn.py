import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv("customer_churn.csv")


data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})
data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})


X = data[["Age", "MonthlyCharges", "Tenure", "Gender"]]
y = data["Churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))


new_customer = [[35, 80, 12, 0]]

result = model.predict(new_customer)

if result[0] == 1:
    print("Customer will churn")
else:
    print("Customer will not churn")