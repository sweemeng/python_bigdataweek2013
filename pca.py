from sklearn import decomposition
import datas


def main():
    training = datas.training
    pca = decomposition.PCA()
    pca.fit(training.ix[:,'Pclass':])
    result =  pca.explained_variance_
    pca.n_components = 3
    reduced = pca.fit_transform(training.ix[:,'Pclass':])
    return reduced

def reduced_data(data):
    pca = decomposition.PCA()
    pca.fit(data.ix[:,'Pclass':])
    result =  pca.explained_variance_
    pca.n_components = 3
    reduced = pca.fit_transform(data.ix[:,'Pclass':])
    return reduced


if __name__ == "__main__":
    main()
