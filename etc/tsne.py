import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


def mnist_dataset():
    digits = datasets.load_digits(n_class=10)
    data = digits.data
    label = digits.target
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_feature(data, label):
    transform = MinMaxScaler(feature_range=(0., 1.))
    transform.fit(data)
    data = transform.transform(data)

    plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 9.),
                 fontdict={'weight': 'bold', 'size': 9})


if __name__ == '__main__':
    data, label, n_samples, n_features = mnist_dataset()
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto")
    result = tsne.fit_transform(data)
    plot_feature(result, label)
    plt.show()
