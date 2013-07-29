import pandas


training = pandas.read_csv("data/train.csv", index_col="PassengerId", parse_dates=False)
testing  = pandas.read_csv("data/test.csv", index_col="PassengerId", parse_dates=False)

def clean_data(data_frame):
    data_frame = data_frame.drop(
        ['Name','Cabin','Ticket','Parch'],axis=1)
    data_frame['Sex'] = data_frame['Sex'].replace(
        ['male','female'], [1,0])
    data_frame['Embarked'] = data_frame['Embarked'].replace(
       ['C','S','Q'], [0,1,2])
    
    data_frame['Age'] = data_frame['Age'].fillna(data_frame['Age'].mean())
    data_frame['Embarked'] = data_frame['Embarked'].fillna(
        data_frame['Embarked'].mean())

    data_frame['Fare'] = data_frame['Fare'].fillna(
        data_frame['Fare'].mean())

    return data_frame

def write_data(series):
    testing = pandas.DataFrame.from_csv("data/test.csv", index_col="PassengerId", parse_dates=False)
    temp = pandas.DataFrame({"Survived":pandas.Series(series, index=testing.index)})
    temp.to_csv("result.csv")

training = clean_data(training)
testing = clean_data(testing)
