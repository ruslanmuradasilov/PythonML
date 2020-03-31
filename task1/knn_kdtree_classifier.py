import numpy as np
from task1.kdtree import Kdtree

# from tensorflow import keras
# from keras.datasets import mnist
from task1.tests import test_knn_model


class KnnKdtreeClassifier(object):
    '''Классификатор реализует взвешенное голосование по ближайшим соседям.
    При подсчете расcтояния используется l1-метрика.
    Поиск ближайшего соседа осуществляется поиском по kd-дереву.
    Параметры
    ----------
    n_neighbors : int, optional
        Число ближайших соседей, учитывающихся в голосовании
    weights : str, optional (default = 'uniform')
        веса, используемые в голосовании. Возможные значения:
        - 'uniform' : все веса равны.
        - 'distance' : веса обратно пропорциональны расстоянию до классифицируемого объекта
        -  функция, которая получает на вход массив расстояний и возвращает массив весов
    leaf_size: int, optional
        Максимально допустимый размер листа дерева
    '''

    def __init__(self, n_neighbors=1, weights='uniform', leaf_size=30):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.leaf_size = leaf_size
        self.n_classes = None
        self.kdtree = None

    def fit(self, x, y):
        '''Обучение модели - построение kd-дерева
        Парметры
        ----------
        x : двумерным массив признаков размера n_queries x n_features
        y : массив/список правильных меток размера n_queries
        Выход
        -------
        Метод возвращает обученную модель
        '''
        self.n_classes = int(np.max(y)) + 1
        self.kdtree = Kdtree(x, y, self.leaf_size, self.weights)
        return self

    def predict(self, x):
        """ Предсказание класса для входных объектов - поиск по kd-дереву ближайших соседей
        и взвешенное голосование
        Параметры
        ----------
        X : двумерным массив признаков размера n_queries x n_features
        Выход
        -------
        y : Массив размера n_queries
        """
        result = list()

        for i in x:
            result.append(self.kdtree.classify_point(i, self.n_neighbors))

        return np.array(result)

    def predict_proba(self, X):
        """Возвращает вероятности классов для входных объектов
        Параметры
        ----------
        X : двумерным массив признаков размера n_queries x n_features
        Выход
        -------
        p : массив размера n_queries x n_classes] c вероятностями принадлежности
        объекта к каждому классу
        """
        tree = self.kdtree
        p = list()

        def classify_k_neighbors(kn):
            classes = np.zeros(self.n_classes)

            max_key, weights_of_classes = tree.voting(kn=kn)
            sum = 0
            for key in weights_of_classes:
                value = weights_of_classes.get(key)
                classes[int(key)] = value
                sum += value
            return np.divide(classes, sum)

        for i in X:
            leaf = tree.closest_leaf(i)
            result_for_current_query = tree.closest_k_neighbors(pivot=i, node=leaf,
                                                                k_neighbors=self.n_neighbors)
            p.append(classify_k_neighbors(result_for_current_query))

        return np.array(p)

    def kneighbors(self, x, n_neighbors):
        """Возвращает n_neighbors ближайших соседей для всех входных объектов при помощи поска по kd-дереву
        и расстояния до них
        Параметры
        ----------
        X : двумерным массив признаков размера n_queries x n_features
        Выход
        -------
        neigh_dist массив размера n_queries х n_neighbors
        расстояния до ближайших элементов
        neigh_indarray, массив размера n_queries x n_neighbors
        индексы ближайших элементов
        """
        neigh_dist = np.zeros((x.shape[0], n_neighbors))
        neigh_indarray = np.zeros((x.shape[0], n_neighbors))

        tree = self.kdtree

        for i, index in zip(x, range(x.shape[0])):
            leaf = tree.closest_leaf(i)
            result_for_current_query = tree.closest_k_neighbors(pivot=i, node=leaf,
                                                                k_neighbors=self.n_neighbors)
            neigh_dist[index] = result_for_current_query[:, -1]
            neigh_indarray[index] = result_for_current_query[:, -2]

        return neigh_dist, neigh_indarray

    def score(self, y_gt, y_pred):
        """Возвращает точность классификации
        Параметры
        ----------
        y_gt : правильные метки объектов
        y_pred: предсказанные метки объектов
        Выход
        -------
        accuracy - точность классификации
        """
        sum = 0
        for i in range(len(y_gt)):
            if y_gt[i] - y_pred[i] == 0:
                sum += 1
        return sum / len(y_gt)


# Test Simple
points = np.float64([(4, 7), (7, 13), (9, 4), (11, 10), (14, 11), (16, 10), (15, 3)])
y = np.float64([0, 2, 0, 0, 1, 3, 0])
pivot = np.float64([[14, 9], ])
knn = KnnKdtreeClassifier(n_neighbors=1, weights='uniform')
knn.fit(points, y)
print(knn.predict(pivot))


# Test MNIST
# (trainX, trainY), (testX, testY) = mnist.load_data()
#
# trainX = trainX.reshape(trainX.shape[0], 784)
# testX = testX.reshape(testX.shape[0], 784)
# trainX = trainX.astype('float32')
# testX = testX.astype('float32')
#
# trainX /= 255
# testX /= 255
#
# x = trainX[:100]
# y = trainY[:100]
# test = testX[:2]
#
# knn = KnnKdtreeClassifier(n_neighbors=3, weights='distance')
# knn.fit(x, y)
# y_pred = knn.predict(test)
# neigh_dist, neigh_indarray = knn.kneighbors(test, 3)
# print(neigh_dist)
# print(neigh_indarray)
# acc = knn.score(testY[:2], y_pred)
# print('Your accuracy is %s' % str(acc))

# Test
# weights=lambda x: np.log2(x)
# model = KnnKdtreeClassifier(n_neighbors=1, weights='uniform')
# test_knn_model(model)