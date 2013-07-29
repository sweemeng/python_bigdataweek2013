import datas
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from pandas import Series, DataFrame
import pca

def main():
    training = datas.training
    training = training.drop(["SibSp"], axis=1)
    testing  = datas.testing
    testing  = testing.drop(["SibSp"], axis=1)
    random_forest = RandomForestClassifier(n_estimators=100, min_samples_split=1, max_depth=None, min_samples_leaf=5)
    random_forest = random_forest.fit(pca.reduced_data(training),training.ix[:,'Survived'])
    result = random_forest.predict(pca.reduced_data(testing))
    return result

def test():
    training = datas.training
    training = training.drop(["SibSp"], axis=1)
    random_forest = RandomForestClassifier(n_estimators=100, min_samples_split=1, max_depth=None, min_samples_leaf=5)
    kfold    = cross_validation.KFold(len(training), 3)
    result   = cross_validation.cross_val_score(random_forest, training.ix[:,'Pclass':], training.ix[:,'Survived'], cv=kfold,n_jobs=1)
    print result

def test_pca():
    training = datas.training
    random_forest = RandomForestClassifier(n_estimators=100, min_samples_split=1, max_depth=None, min_samples_leaf=5)
    kfold    = cross_validation.KFold(len(training), 3)
    result   = cross_validation.cross_val_score(random_forest, pca.main(), training.ix[:,'Survived'], cv=kfold,n_jobs=1)
    print result

if __name__ == "__main__":
    result = main()

    print result
    testing = DataFrame.from_csv("data/test.csv", index_col="PassengerId", parse_dates=False)
    print testing
    temp = DataFrame({"Survived":Series(result, index=testing.index)})
    temp.to_csv("result.csv")

