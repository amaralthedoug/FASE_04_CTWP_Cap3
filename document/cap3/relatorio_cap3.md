Relatório Final: Classificação de Variedades de Trigo utilizando Machine Learning
Cap 3 – (IR ALÉM) Implementando Algoritmos de Machine Learning com Scikit-learn
Autores: Douglas Rafael do Amaral, Cláudio Sartori, William Albert Cesário Vasconcelos, Pedro Alves da Silva
Data: 27 de Novembro de 2025
1. Introdução e Objetivo

Este relatório apresenta a aplicação da metodologia CRISP-DM para desenvolver um modelo de Machine Learning capaz de classificar três variedades de grãos de trigo (Kama, Rosa e Canadian). O objetivo é demonstrar como a automação desse processo pode aumentar a eficiência e reduzir erros em cooperativas agrícolas de pequeno porte, onde a classificação costuma ser feita manualmente por especialistas.

O estudo utiliza o Seeds Dataset, disponibilizado pelo UCI Machine Learning Repository [1], composto por 210 amostras e 7 atributos geométricos que descrevem propriedades físicas dos grãos.

2. Metodologia (CRISP-DM)

A execução do projeto seguiu as quatro macroetapas especificadas na atividade: análise e pré-processamento dos dados, implementação dos modelos, otimização via hiperparâmetros e interpretação dos resultados obtidos.

2.1. Análise e Pré-processamento dos Dados

As etapas realizadas foram:

Carregamento e compreensão dos dados: O dataset possui 210 instâncias e 7 atributos contínuos (Área, Perímetro, Compacidade, Comprimento do Núcleo, Largura do Núcleo, Coeficiente de Assimetria e Comprimento do Sulco do Núcleo).

Estatísticas descritivas: O cálculo de média, mínimo, máximo, desvio-padrão e quartis permitiu entender a distribuição de cada atributo.

Valores ausentes: A verificação confirmou que não existem dados faltantes, eliminando a necessidade de imputação.

Visualizações: Foram gerados:

histogramas para todas as features;

boxplots para detecção de outliers;

matriz de correlação para análise das relações entre variáveis;

scatter plots (pairplot) para visualizar relações entre pares de atributos.

Escalonamento das features: Como modelos como KNN e SVM são sensíveis à escala, foi aplicado o StandardScaler, centralizando os atributos e padronizando seus valores (média 0, desvio-padrão 1).

Divisão dos dados: O conjunto foi dividido em 70% para treino e 30% para teste, utilizando estratificação para manter a proporção das classes.

2.2. Implementação e Comparação dos Modelos

Foram treinados cinco algoritmos de classificação:

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Random Forest

Logistic Regression

Gaussian Naive Bayes

Cada modelo foi avaliado utilizando as métricas:
Acurácia, Precisão, Recall, F1-Score e Matriz de Confusão.

Os resultados iniciais mostraram desempenhos competitivos, com destaque para o Random Forest.

2.3. Otimização dos Modelos

Os modelos com hiperparâmetros relevantes foram otimizados por meio de Grid Search com Cross-Validation (cv=5):

KNN

SVM

Random Forest

Logistic Regression

O Gaussian Naive Bayes não foi otimizado por não possuir hiperparâmetros ajustáveis relevantes.

Após o processo, os modelos foram reavaliados no conjunto de teste utilizando as melhores combinações de parâmetros encontradas.

3. Resultados e Interpretação
3.1. Comparação de Desempenho

A tabela abaixo resume o desempenho dos modelos antes e depois da otimização:

Modelo	Otimizacao	Acuracia	Precisao	Recall	F1-Score	Melhores Parametros
KNN	Inicial	0.8730	0.8721	0.8730	0.8713	N/A
KNN	Otimizado	0.8889	0.8880	0.8889	0.8881	{'n_neighbors': 9, 'weights': 'uniform'}
SVM	Inicial	0.8730	0.8721	0.8730	0.8713	N/A
SVM	Otimizado	0.8730	0.8755	0.8730	0.8729	{'C': 10, 'kernel': 'linear'}
RandomForest	Inicial	0.9206	0.9239	0.9206	0.9192	N/A
RandomForest	Otimizado	0.9365	0.9432	0.9365	0.9349	{'max_depth': None, 'n_estimators': 50}
LogisticRegression	Inicial	0.8571	0.8571	0.8571	0.8543	N/A
LogisticRegression	Otimizado	0.8889	0.8899	0.8889	0.8884	{'C': 10}
NaiveBayes	Inicial	0.8254	0.8339	0.8254	0.8251	N/A
NaiveBayes	Otimizado	0.8254	0.8339	0.8254	0.8251	N/A
3.2. Principais Insights

Modelo com melhor desempenho:
O Random Forest Otimizado atingiu 93,65% de acurácia, sendo o modelo mais eficiente para este problema.

Impacto da otimização:

Random Forest e Logistic Regression apresentaram melhorias significativas após o ajuste dos hiperparâmetros.

O desempenho do SVM manteve-se estável, sugerindo que sua performance já era próxima do ótimo.

Natureza do problema:
O forte desempenho de modelos baseados em árvores e métodos baseados em distância indica que a separação entre as variedades é não linear.

Aplicabilidade prática:
Um modelo com acurácia superior a 93% reduz drasticamente o erro humano e demonstra viabilidade real para uso em cooperativas agrícolas como sistema automatizado de apoio à decisão.

4. Conclusão

O pipeline de Machine Learning desenvolvido cumpriu com sucesso o objetivo de classificar as três variedades de trigo presentes no Seeds Dataset. O modelo Random Forest Otimizado apresentou o melhor desempenho e pode ser considerado a escolha recomendada para uso operacional.

A atividade contemplou todas as etapas propostas no Capítulo 3: análise dos dados, implementação de múltiplos modelos, comparação de métricas, otimização dos hiperparâmetros e interpretação final dos resultados. Os próximos passos para evolução do projeto incluem a criação de uma interface de produção e a integração com visão computacional para processamento automático de imagens dos grãos.

5. Referências

[1] Charytanowicz, M., Niewczas, J., Kulczycki, P., Kowalski, P., & Lukasik, S. (2010). Seeds [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5H30K