import matplotlib
matplotlib.use('Agg')  # Use backend não interativo para evitar avisos de GUI

from flask import Flask, render_template, request, jsonify
import io
import matplotlib.pyplot as plt
import numpy as np
import base64
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, confusion_matrix, calinski_harabasz_score)

app = Flask(__name__)

# Variáveis globais para KNN
knn_model = None
X_train, X_test, y_train, y_test = None, None, None, None

@app.route('/')
def home():
    return render_template('front.html')

# Rotas para KNN
@app.route('/train_knn', methods=['POST'])
def train_knn():
    global knn_model, X_train, X_test, y_train, y_test
    # Carregar o dataset Iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Dividir em treino e teste (70% treino, 30% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Criar e treinar o classificador KNN (usando os 4 atributos)
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)

    return jsonify({"message": "Treinamento do KNN concluído"})

@app.route('/test_knn', methods=['GET'])
def test_knn():
    global knn_model, X_test, y_test, X_train, y_train
    if knn_model is None:
        return jsonify({"error": "Modelo KNN não treinado"}), 400

    # --- Cálculo das métricas para o conjunto de teste ---
    y_pred_test = knn_model.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    rec_test = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    metrics_test = {"accuracy": round(acc_test, 3), "precision": round(prec_test, 3), "recall": round(rec_test, 3)}

    # --- Cálculo das métricas para o conjunto de treinamento ---
    y_pred_train = knn_model.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    prec_train = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
    rec_train = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
    metrics_train = {"accuracy": round(acc_train, 3), "precision": round(prec_train, 3), "recall": round(rec_train, 3)}

    # --- Gerar a matriz de confusão (teste) para visualização ---
    plt.figure()
    cm_test = confusion_matrix(y_test, y_pred_test)
    plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão (Teste)')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    classes = load_iris().target_names
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    
    buf_cm = io.BytesIO()
    plt.savefig(buf_cm, format='png')
    buf_cm.seek(0)
    plt.close()
    cm_img = base64.b64encode(buf_cm.getvalue()).decode('utf-8')
    
    # --- Gerar a superfície de decisão (utilizando apenas os atributos 2 e 3 para o teste) ---
    iris = load_iris()
    X_train_2 = X_train[:, 2:4]
    X_test_2 = X_test[:, 2:4]
    model2 = KNeighborsClassifier(n_neighbors=3)
    model2.fit(X_train_2, y_train)
    
    x_min, x_max = X_train_2[:, 0].min() - 1, X_train_2[:, 0].max() + 1
    y_min, y_max = X_train_2[:, 1].min() - 1, X_train_2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    for i, target_name in enumerate(iris.target_names):
        idx = np.where(y_test == i)
        plt.scatter(X_test_2[idx, 0], X_test_2[idx, 1], edgecolors='k', label=target_name)
    plt.xlabel('Comprimento da Pétala (cm)')
    plt.ylabel('Largura da Pétala (cm)')
    plt.title('Superfície de Decisão (Teste)')
    plt.legend()
    
    buf_ds = io.BytesIO()
    plt.savefig(buf_ds, format='png')
    buf_ds.seek(0)
    plt.close()
    ds_img = base64.b64encode(buf_ds.getvalue()).decode('utf-8')
    
    # Retorna todos os resultados
    return jsonify({
        "test_metrics": metrics_test,
        "train_metrics": metrics_train,
        "confusion_matrix": cm_img,
        "decision_surface": ds_img
    })

@app.route('/predict', methods=['POST'])
def predict():
    global knn_model, X_train, y_train
    if knn_model is None:
        return jsonify({"error": "Modelo KNN não treinado"}), 400
    data = request.json
    try:
        values = [float(data[key]) for key in ["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    except Exception as e:
        return jsonify({"error": "Dados inválidos"}), 400

    pred = knn_model.predict([values])[0]
    iris = load_iris()
    result = iris.target_names[pred]
    # Calcula a acurácia do modelo no conjunto de treino
    acc_train = knn_model.score(X_train, y_train)
    # Retorna o resultado com a acurácia formatada
    return jsonify({"predicao": f"{result} (ACC: {round(acc_train*100, 0)}%)"})

# Variáveis globais para K-means
kmeans_model = None
X_kmeans = None

# Rotas para K-means
@app.route('/train_kmeans', methods=['POST'])
def train_kmeans():
    global kmeans_model, X_kmeans
    # Carregar o dataset Iris
    iris = load_iris()
    X_kmeans = iris.data[:, 2:4]  # Usando comprimento e largura da pétala para visualização
    
    # Criar e treinar o modelo K-means
    kmeans_model = KMeans(n_clusters=3, random_state=50, n_init=10)
    kmeans_model.fit(X_kmeans)
    
    return jsonify({"message": "Treinamento do K-means concluído"})

@app.route('/test_kmeans', methods=['GET'])
def test_kmeans():
    global kmeans_model, X_kmeans
    if kmeans_model is None:
        return jsonify({"error": "Modelo K-means não treinado"}), 400
    
    # Obter labels e centróides
    labels = kmeans_model.labels_
    centroids = kmeans_model.cluster_centers_
    
    # Calcular métricas
    calinski_harabasz = calinski_harabasz_score(X_kmeans, labels)
    inertia = kmeans_model.inertia_
    n_iter = kmeans_model.n_iter_
    
    # Plotar gráfico de clusters
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_kmeans[:, 0], X_kmeans[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centróides')
    plt.xlabel('Comprimento da Pétala (cm)')
    plt.ylabel('Largura da Pétala (cm)')
    plt.title('Clusters encontrados pelo K-means')
    plt.legend()
    
    
    # Separar os gráficos para enviar individualmente
    # Gráfico de clusters
    plt.figure()
    plt.scatter(X_kmeans[:, 0], X_kmeans[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centróides')
    plt.xlabel('Comprimento da Pétala (cm)')
    plt.ylabel('Largura da Pétala (cm)')
    plt.title('Clusters encontrados pelo K-means')
    plt.legend()
    buf_cluster = io.BytesIO()
    plt.savefig(buf_cluster, format='png')
    buf_cluster.seek(0)
    plt.close()
    cluster_img = base64.b64encode(buf_cluster.getvalue()).decode('utf-8')

    # Gráfico comparativo entre rótulos reais e clusters
    iris = load_iris()
    true_labels = iris.target
    plt.figure()
    plt.scatter(X_kmeans[:, 0], X_kmeans[:, 1], c=true_labels, cmap='tab10', alpha=0.7)
    plt.xlabel('Comprimento da Pétala (cm)')
    plt.ylabel('Largura da Pétala (cm)')
    plt.title('Comparação com Rótulos Originais')
    buf_compare = io.BytesIO()
    plt.savefig(buf_compare, format='png')
    buf_compare.seek(0)
    plt.close()
    compare_img = base64.b64encode(buf_compare.getvalue()).decode('utf-8')

    
    return jsonify({
        "calinski_harabasz_score": calinski_harabasz,
        "inertia": inertia,
        "n_iter": n_iter,
        "cluster_plot": cluster_img,
        "comparison_plot": compare_img
})

if __name__ == '__main__':
    app.run(debug=True)