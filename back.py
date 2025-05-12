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
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

app = Flask(__name__)

model = None
X_train, X_test, y_train, y_test = None, None, None, None

@app.route('/')
def home():
    return render_template('front.html')

@app.route('/train', methods=['POST'])
def train():
    global model, X_train, X_test, y_train, y_test
    # Carregar o dataset Iris
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Dividir em treino e teste (70% treino, 30% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Criar e treinar o classificador KNN (usando os 4 atributos)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    return jsonify({"message": "Treinamento concluído"})

@app.route('/test', methods=['GET'])
def test():
    global model, X_test, y_test, X_train, y_train
    if model is None:
        return jsonify({"error": "Modelo não treinado"}), 400

    # --- Cálculo das métricas para o conjunto de teste ---
    y_pred_test = model.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    rec_test = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    metrics_test = {"accuracy": round(acc_test, 3), "precision": round(prec_test, 3), "recall": round(rec_test, 3)}

    # Cálculo manual a partir da matriz de confusão (teste)
    cm_test = confusion_matrix(y_test, y_pred_test)
    accuracy_manual_test = np.trace(cm_test) / np.sum(cm_test)
    precisions_test = []
    recalls_test = []
    supports_test = []
    for i in range(len(cm_test)):
        TP = cm_test[i, i]
        FP = np.sum(cm_test[:, i]) - TP
        FN = np.sum(cm_test[i, :]) - TP
        precisions_test.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
        recalls_test.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        supports_test.append(np.sum(cm_test[i, :]))
    precision_manual_test = np.average(precisions_test, weights=supports_test)
    recall_manual_test = np.average(recalls_test, weights=supports_test)
    manual_metrics_test = {
        "accuracy_manual": round(accuracy_manual_test, 3),
        "precision_manual": round(precision_manual_test, 3),
        "recall_manual": round(recall_manual_test, 3)
    }

    # --- Cálculo das métricas para o conjunto de treinamento ---
    y_pred_train = model.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    prec_train = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
    rec_train = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
    metrics_train = {"accuracy": round(acc_train, 3), "precision": round(prec_train, 3), "recall": round(rec_train, 3)}

    # Cálculo manual a partir da matriz de confusão (treino)
    cm_train = confusion_matrix(y_train, y_pred_train)
    accuracy_manual_train = np.trace(cm_train) / np.sum(cm_train)
    precisions_train = []
    recalls_train = []
    supports_train = []
    for i in range(len(cm_train)):
        TP = cm_train[i, i]
        FP = np.sum(cm_train[:, i]) - TP
        FN = np.sum(cm_train[i, :]) - TP
        precisions_train.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
        recalls_train.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        supports_train.append(np.sum(cm_train[i, :]))
    precision_manual_train = np.average(precisions_train, weights=supports_train)
    recall_manual_train = np.average(recalls_train, weights=supports_train)
    manual_metrics_train = {
        "accuracy_manual": round(accuracy_manual_train, 3),
        "precision_manual": round(precision_manual_train, 3),
        "recall_manual": round(recall_manual_train, 3)
    }

    # --- Gerar a matriz de confusão (teste) para visualização ---
    plt.figure()
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
        "manual_test_metrics": manual_metrics_test,
        "train_metrics": metrics_train,
        "manual_train_metrics": manual_metrics_train,
        "confusion_matrix": cm_img,
        "decision_surface": ds_img
    })


@app.route('/predict', methods=['POST'])
def predict():
    global model, X_train, y_train
    if model is None:
        return jsonify({"error": "Modelo não treinado"}), 400
    data = request.json
    try:
        values = [float(data[key]) for key in ["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    except Exception as e:
        return jsonify({"error": "Dados inválidos"}), 400

    pred = model.predict([values])[0]
    iris = load_iris()
    result = iris.target_names[pred]
    # Calcula a acurácia do modelo no conjunto de treino
    acc_train = model.score(X_train, y_train)
    # Retorna o resultado com a acurácia formatada
    return jsonify({"predicao": f"{result} (ACC: {round(acc_train*100, 0)}%)"})


if __name__ == '__main__':
    app.run(debug=True)
