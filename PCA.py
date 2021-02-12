from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Cargamos los datos de ejemplo de la base de datos Iris
iris = load_iris()
X = iris.data
y = iris.target

# Creamos un objeto PCA con dos componentes principales
pca = PCA(n_components=2)

# Ajustamos el modelo a los datos
pca.fit(X)

# Transformamos los datos en las dos componentes principales
X_pca = pca.transform(X)

# Graficamos los datos en las dos componentes principales
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()