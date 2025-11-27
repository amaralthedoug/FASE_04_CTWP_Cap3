FIAP – Faculdade de Informática e Administração Paulista
Cap 3 – IR ALÉM: Implementando Algoritmos de Machine Learning com Scikit-learn
Da Terra ao Código: Classificação Automatizada de Grãos de Trigo
Nome do Projeto

Classificação de Variedades de Trigo com Machine Learning (Seeds Dataset)

Nome do Grupo

Grupo CAP3 – Classificação de Grãos

👨‍🎓 Integrantes

William Albert Cesário Vasconcelos – contact@williamvasconcelos.com

Pedro Alves da Silva – pedro19993613@gmail.com

Douglas Rafael do Amaral – douglas.rafa.amaral@gmail.com

Cláudio Sartori – csartorirp@gmail.com

👩‍🏫 Professores

Tutor(a):
Coordenador(a): André Godoi

📜 Descrição do Projeto

Este projeto tem como objetivo aplicar a metodologia CRISP-DM para desenvolver um sistema completo de classificação de grãos de trigo, automatizando um processo que, em cooperativas agrícolas de pequeno porte, é tradicionalmente realizado de forma manual e sujeito a erros.

Utilizando o Seeds Dataset (UCI Machine Learning Repository), que contém 210 amostras de três variedades de trigo (Kama, Rosa e Canadian), foram analisadas sete características morfológicas dos grãos, como área, perímetro, compacidade e comprimento do sulco.

O trabalho foi estruturado em quatro etapas principais:

1. Análise e Pré-processamento dos Dados

Os dados foram carregados, explorados e descritos estatisticamente. Foram gerados histogramas, boxplots e matriz de correlação para identificar padrões, distribuições e possíveis outliers.
Como os algoritmos são sensíveis à escala dos atributos, aplicou-se a técnica de StandardScaler, garantindo que todas as variáveis tivessem média 0 e desvio padrão 1.
Os dados foram divididos em treino (70%) e teste (30%) usando amostragem estratificada.

2. Implementação e Comparação de Modelos

Foram utilizados cinco algoritmos de classificação:
K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest, Logistic Regression e Naive Bayes.
Cada modelo foi treinado com o conjunto de treino e avaliado com o conjunto de teste utilizando métricas como Acurácia, Precisão, Recall, F1-Score e matriz de confusão.

3. Otimização dos Modelos

Os modelos foram otimizados usando Grid Search com validação cruzada (5-fold) para identificar os melhores hiperparâmetros.
Os resultados mostraram ganhos de desempenho principalmente para KNN, Logistic Regression e Random Forest.

4. Interpretação dos Resultados

O Random Forest Otimizado apresentou o melhor desempenho geral, alcançando 93,65% de acurácia, se destacando como o modelo ideal para aplicação prática em cooperativas agrícolas.
Essa precisão torna o processo de classificação mais rápido, menos sujeito a falhas humanas e escalável para produção.

O projeto demonstra de forma clara como métodos de aprendizado de máquina podem apoiar processos agrícolas e otimizar operações de classificação.

📁 Estrutura de Pastas

A organização segue as boas práticas recomendadas para projetos acadêmicos e técnicos:

.github/
Arquivos de configuração e automação relacionados ao GitHub.

assets/
Contém imagens, gráficos e demais arquivos não estruturados utilizados no projeto.
Dentro de assets/cap3/ ficam os gráficos gerados automaticamente pelo pipeline.

config/
Arquivos de configuração para ajustes internos do projeto (opcional nesta fase).

data/
Contém o dataset original utilizado pelo projeto (seeds_dataset.txt).

document/
Relatórios e documentação final do projeto.
Em document/cap3/ ficam relatórios, tabelas comparativas e outputs do treinamento.

scripts/
Espaço reservado para scripts auxiliares (ex.: automação, deploy, backup).

src/
Código-fonte do projeto. Em src/cap3/ encontra-se o script principal seeds_classifier.py.

README.md
Arquivo atual, contendo todas as explicações gerais e instruções do projeto.

requirements.txt
Lista das dependências Python necessárias para execução do projeto.

🔧 Como Executar o Código
Pré-requisitos

Python 3.12 ou superior

Ambiente virtual (recomendado)

Dependências presentes em requirements.txt

Passo a passo

Clone o repositório:

git clone <link-do-repositorio>

Entre no diretório:

cd FASE_04_CTWP_Cap3

Crie um ambiente virtual:

python3 -m venv .venv

Ative o ambiente:

source .venv/bin/activate (Linux/Mac)

.venv\Scripts\activate (Windows)

Instale as dependências:

pip install -r requirements.txt

Execute o script:

python3 src/cap3/seeds_classifier.py

Os gráficos, relatórios e tabelas serão automaticamente gerados nas pastas assets/cap3 e document/cap3.
