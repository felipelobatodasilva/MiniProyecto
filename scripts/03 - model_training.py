import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def plot_training_history(model, X_val, y_val):
    """
    Plota o histórico de treinamento do modelo MLP.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_, label='Pérdida de Entrenamiento')
    if hasattr(model, 'validation_scores_'):
        plt.plot(model.validation_scores_, label='Puntuación de Validación')
    plt.title('Historial de Entrenamiento')
    plt.xlabel('Iteraciones')
    plt.ylabel('Pérdida / Puntuación')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_proba):
    """
    Plota a curva ROC.    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.savefig('roc_curve.png')
    plt.close()

def plot_pr_curve(y_true, y_pred_proba):
    """
    Plota a curva Precision-Recall.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision)
    plt.xlabel('Exhaustividad (Recall)')
    plt.ylabel('Precisión')
    plt.title('Curva Precisión-Exhaustividad')
    plt.savefig('pr_curve.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """
    Plota a matriz de confusão.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.savefig('confusion_matrix.png')
    plt.close()

def train_evaluate_model(model, X, y):
    """
    Treina e avalia um modelo usando validação cruzada.
    Retorna as métricas ROC AUC e PR AUC.
    """
    # Separar dados para validação
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar modelo
    model.fit(X_train, y_train)
    
    # Gerar predições
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    # Gerar visualizações
    plot_training_history(model, X_val, y_val)
    plot_roc_curve(y_val, y_pred_proba)
    plot_pr_curve(y_val, y_pred_proba)
    plot_confusion_matrix(y_val, y_pred)
    
    # Calcular métricas usando validação cruzada
    cv_results = cross_validate(
        model, X, y,
        cv=5,
        scoring={'roc_auc': 'roc_auc', 'average_precision': 'average_precision'},
        return_train_score=True
    )
    
    roc_auc = cv_results['test_roc_auc'].mean()
    pr_auc = cv_results['test_average_precision'].mean()
    
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    return cv_results

def generate_submission(model, X_train, y_train, X_test):
    """
    Treina o modelo em todos os dados de treino e gera predições para o conjunto de teste.
    """
    print("\nTreinando MLP em todos os dados de treino...")
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Criar DataFrame de submissão com os IDs corretos
    submission = pd.DataFrame({
        'ID': test_data['ID'],
        'Bankruptcy': y_pred_proba
    })
    
    filename = 'submission.csv'
    submission.to_csv(filename, index=False)
    print(f"Arquivo de submissão salvo como {filename}")
    
    return submission

if __name__ == "__main__":
    # Carregar dados processados
    print("Cargando datos procesados...")
    X_train = np.load('X_train_processed.npy')
    y_train = np.load('y_train_processed.npy')
    X_test = np.load('X_test_processed.npy')
    
    # Carregar dados originais para IDs
    test_data = pd.read_csv('../archivos/test_data.csv')
    sample_submission = pd.read_csv('../archivos/sampleSubmission.csv')
    
    # Definir modelos
    print("\nDefiniendo modelos...")
    models = {
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42,
            early_stopping=True
        )
    }
    
    # Treinar e avaliar modelos
    print("\nEntrenando y evaluando modelos...")
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\nEvaluando {name}...")
        cv_results = train_evaluate_model(model, X_train, y_train)
        
        if cv_results['test_roc_auc'].mean() > best_score:
            best_score = cv_results['test_roc_auc'].mean()
            best_model = model
    
    # Gerar submissão com melhor modelo
    print("\nGenerando archivo de submission...")
    generate_submission(best_model, X_train, y_train, X_test)
    
    print("\nProceso completado exitosamente!") 