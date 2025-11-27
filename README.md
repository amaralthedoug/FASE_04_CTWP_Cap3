# FIAP – Faculdade de Informática e Administração Paulista  
## Cap 3 – IR ALÉM: Implementando Algoritmos de Machine Learning com Scikit-learn  
### *Da Terra ao Código: Classificação Automatizada de Grãos de Trigo*

---

**Classificação de Variedades de Trigo com Machine Learning (Seeds Dataset)**
---
**Grupo CAP3 – Classificação de Grãos**

---

## 👨‍🎓 Integrantes  
- **William Albert Cesário Vasconcelos** – contact@williamvasconcelos.com  
- **Pedro Alves da Silva** – pedro19993613@gmail.com  
- **Douglas Rafael do Amaral** – douglas.rafa.amaral@gmail.com  
- **Cláudio Sartori** – csartorirp@gmail.com  

---

## 👩‍🏫 Professores  
**Tutor(a) / Coordenador(a): André Godoi**

---

# 📜 Descrição do Projeto  

Este projeto tem como objetivo aplicar a metodologia **CRISP-DM** para desenvolver um sistema completo de **classificação de grãos de trigo**, automatizando um processo que normalmente é realizado de forma manual em cooperativas agrícolas de pequeno porte — tornando-o mais rápido, preciso e menos sujeito a erros.

Utilizando o **Seeds Dataset (UCI Machine Learning Repository)**, contendo **210 amostras** de três variedades (Kama, Rosa e Canadian), foram analisadas sete características morfológicas dos grãos, incluindo:

- Área  
- Perímetro  
- Compacidade  
- Comprimento do núcleo  
- Largura do núcleo  
- Coeficiente de assimetria  
- Comprimento do sulco  

---

# 🧭 Etapas do Trabalho (CRISP-DM)

## **1. Análise e Pré-processamento dos Dados**
- Carregamento e exploração do dataset  
- Geração de histogramas, boxplots e matriz de correlação  
- Identificação de padrões, distribuições e possíveis outliers  
- Aplicação do **StandardScaler** para padronizar as variáveis  
- Divisão treino/teste (70% / 30%) com amostragem estratificada  

---

## **2. Implementação e Comparação de Modelos**
Algoritmos utilizados:  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Random Forest  
- Logistic Regression  
- Naive Bayes (GaussianNB)

Cada modelo foi avaliado utilizando:  
- **Acurácia**  
- **Precisão**  
- **Recall**  
- **F1-Score**  
- **Matriz de confusão**

---

## **3. Otimização dos Modelos**
- Aplicação de **Grid Search (5-fold cross validation)**  
- Melhora significativa nos modelos KNN, Random Forest e Logistic Regression  

---

## **4. Interpretação dos Resultados**
O modelo com melhor desempenho foi:

### 🏆 **Random Forest Otimizado**  
- **Acurácia:** 93,65%  
- Hiperparâmetros encontrados:  
  - `n_estimators = 50`  
  - `max_depth = None`  

Este modelo mostrou ser ideal para aplicação prática no cenário agrícola, oferecendo maior robustez e precisão na classificação automática de grãos.

---

# 📁 Estrutura de Pastas

