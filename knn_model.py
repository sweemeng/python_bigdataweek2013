from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
import datas
import pca

def test():
    training = datas.training
    knn = KNeighborsClassifier(25, weights="uniform")
    kfold = cross_validation.KFold(len(training), 3)
    result = cross_validation.cross_val_score(knn, pca.reduced_data(training), training['Survived'], cv=kfold,n_jobs=1)
    print result

def main():
    training = datas.training
    training = training.drop(['Embarked'],axis=1)
    testing = datas.testing
    testing = testing.drop(['Embarked'],axis=1)

    knn = KNeighborsClassifier(15,weights="uniform")
    knn.fit(pca.reduced_data(training), training.ix[:,'Survived'])
    result = knn.predict(pca.reduced_data(testing))
    datas.write_data(result)



if __name__ == "__main__":
    main()
