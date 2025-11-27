import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# --- Configuracao de caminhos ---
# Arquivo do dataset (rodando o script a partir da raiz do projeto)
DATA_FILE = "data/seeds_dataset.txt"

# Diretorios de saida para seguir o padrao FIAP
FIG_DIR = "assets/cap3"      # imagens, graficos etc.
DOC_DIR = "document/cap3"    # relatorios, tabelas

# Dicionario de modelos que serao treinados e comparados
MODELS_TO_TEST = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "NaiveBayes": GaussianNB(),
}

# Grades de hiperparametros para otimizacao com GridSearch
PARAM_GRIDS = {
    "KNN": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
    "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
    "NaiveBayes": {},  # GaussianNB nao possui hiperparametros principais para tunar
}

# Nomes das colunas com base na documentacao do UCI
COLUMNS = [
    "Area",
    "Perimeter",
    "Compactness",
    "Length_Kernel",
    "Width_Kernel",
    "Asymmetry_Coeff",
    "Length_Kernel_Groove",
    "Variety",
]

# Garante que os diretorios existam
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DOC_DIR, exist_ok=True)


def load_data(file_path: str) -> pd.DataFrame:
    """Carrega o dataset seeds, separado por espaco e sem cabecalho."""
    print(f"Loading data from {file_path}...")
    # O arquivo usa multiplos espacos como separador, por isso usamos regex no sep
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        engine="python",
    )
    return df


def analyze_and_preprocess(df: pd.DataFrame):
    """Realiza analise e pre processamento inicial dos dados."""
    print("\n--- 1. Data Analysis and Preprocessing ---")

    # Exibe as primeiras linhas do dataset
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    # Calcula estatisticas descritivas
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Verifica valores ausentes
    print("\nMissing values check:")
    print(df.isnull().sum())

    # Visualizacoes exploratorias
    print("\nGenerating visualizations...")

    # Distribuicao das features (histogramas)
    df.drop("Variety", axis=1).hist(figsize=(12, 8))
    plt.suptitle("Feature Distributions (Histograms)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "histograms.png"))
    plt.close()

    # Boxplots para observar distribuicoes e possiveis outliers
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df.drop("Variety", axis=1))
    plt.title("Feature Distributions (Boxplots)")
    plt.savefig(os.path.join(FIG_DIR, "boxplots.png"))
    plt.close()

    # Matriz de correlacao
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df.drop("Variety", axis=1).corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Correlation Matrix of Features")
    plt.savefig(os.path.join(FIG_DIR, "correlation_matrix.png"))
    plt.close()

    # Separa features (X) e alvo (y)
    X = df.drop("Variety", axis=1)
    y = df["Variety"]

    # Escalonamento padronizado das features
    print("\nScaling features using StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y


def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series):
    """Treina e avalia os modelos de classificacao."""
    print("\n--- 2. Model Implementation and Comparison ---")

    # Split treino / teste com estratificacao
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    results = {}

    for name, model in MODELS_TO_TEST.items():
        print(f"\nTraining {name}...")

        # Treinamento do modelo
        model.fit(X_train, y_train)

        # Predicoes no conjunto de teste
        y_pred = model.predict(X_test)

        # Metricas de avaliacao
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "report": classification_report(
                y_test, y_pred, zero_division=0
            ),
        }

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")

        # Matriz de confusao
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=np.unique(y),
            yticklabels=np.unique(y),
        )
        plt.title(f"Confusion Matrix for {name}")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(os.path.join(FIG_DIR, f"cm_{name}.png"))
        plt.close()

    return results, X_train, X_test, y_train, y_test


def optimize_models(
    results: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y: pd.Series,
):
    """Otimiza os modelos usando Grid Search."""
    print("\n--- 3. Model Optimization (Hyperparameter Tuning) ---")

    optimized_results = {}

    for name, model_info in results.items():
        model = model_info["model"]
        param_grid = PARAM_GRIDS[name]

        if not param_grid:
            print(f"Skipping optimization for {name} (no hyperparameters to tune).")
            optimized_results[name] = model_info
            continue

        print(f"\nOptimizing {name} with Grid Search...")

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Reavalia o melhor modelo
        y_pred_opt = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred_opt)
        precision = precision_score(
            y_test, y_pred_opt, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred_opt, average="weighted", zero_division=0
        )
        f1 = f1_score(y_test, y_pred_opt, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred_opt)

        optimized_results[name] = {
            "model": best_model,
            "best_params": grid_search.best_params_,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "report": classification_report(
                y_test, y_pred_opt, zero_division=0
            ),
        }

        print(f"  Best Parameters: {grid_search.best_params_}")
        print(f"  Optimized Accuracy: {accuracy:.4f}")
        print(f"  Improvement: {accuracy - model_info['accuracy']:.4f}")

        # Matriz de confusao otimizada
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=np.unique(y),
            yticklabels=np.unique(y),
        )
        plt.title(f"Optimized Confusion Matrix for {name}")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(os.path.join(FIG_DIR, f"cm_optimized_{name}.png"))
        plt.close()

    return optimized_results


def main():
    """Funcao principal que executa todo o fluxo de machine learning."""

    # 1. Carrega e pre processa os dados
    df = load_data(DATA_FILE)
    X_scaled, y = analyze_and_preprocess(df)

    # Split fixo para avaliacao fora do loop de otimizacao
    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # 2. Treina e avalia modelos iniciais
    initial_results, X_train, _, y_train, _ = train_and_evaluate_models(
        X_scaled, y
    )

    # 3. Otimiza os modelos
    optimized_results = optimize_models(
        initial_results, X_train, y_train, X_test, y_test, y
    )

    # 4. Consolida resultados e gera insights
    print("\n--- 4. Final Results and Interpretation ---")

    comparison_data = []
    for name in initial_results.keys():
        initial = initial_results[name]
        comparison_data.append(
            {
                "Model": name,
                "Optimization": "Initial",
                "Accuracy": initial["accuracy"],
                "Precision": initial["precision"],
                "Recall": initial["recall"],
                "F1-Score": initial["f1_score"],
                "Best Params": "N/A",
            }
        )

        optimized = optimized_results[name]
        comparison_data.append(
            {
                "Model": name,
                "Optimization": "Optimized",
                "Accuracy": optimized["accuracy"],
                "Precision": optimized["precision"],
                "Recall": optimized["recall"],
                "F1-Score": optimized["f1_score"],
                "Best Params": optimized.get("best_params", "N/A"),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    # Salva a tabela de comparacao em formato markdown
    comparison_table_path = os.path.join(
        DOC_DIR, "Model Performance Comparison.md"
    )
    with open(comparison_table_path, "w") as f:
        f.write("## Model Performance Comparison\n\n")
        f.write(comparison_df.to_markdown(index=False, floatfmt=".4f"))

    print(f"\nModel comparison table saved to {comparison_table_path}")
    print("\nSummary of Optimized Model Performance:")
    print(
        comparison_df[
            comparison_df["Optimization"] == "Optimized"
        ].to_markdown(index=False, floatfmt=".4f")
    )

    # Salva todos os relatorios de classificacao
    reports_path = os.path.join(DOC_DIR, "classification_reports.txt")
    with open(reports_path, "w") as f:
        f.write("--- Initial Model Classification Reports ---\n\n")
        for name, res in initial_results.items():
            f.write(f"Model: {name}\n")
            f.write(res["report"])
            f.write("\n" + "=" * 50 + "\n")

        f.write("\n--- Optimized Model Classification Reports ---\n\n")
        for name, res in optimized_results.items():
            f.write(
                f"Model: {name} (Best Params: {res.get('best_params', 'N/A')})\n"
            )
            f.write(res["report"])
            f.write("\n" + "=" * 50 + "\n")

    print(f"\nAll classification reports saved to {reports_path}")

    # Identifica o melhor modelo otimizado
    best_model_row = (
        comparison_df[comparison_df["Optimization"] == "Optimized"]
        .sort_values(by="Accuracy", ascending=False)
        .iloc[0]
    )

    print(
        f"\nBest Performing Model (Optimized): "
        f"{best_model_row['Model']} with Accuracy: {best_model_row['Accuracy']:.4f}"
    )

    # Opcional: salvar o melhor modelo em disco
    # import joblib
    # joblib.dump(
    #     optimized_results[best_model_row["Model"]]["model"],
    #     os.path.join(DOC_DIR, "best_model.pkl"),
    # )


if __name__ == "__main__":
    main()
