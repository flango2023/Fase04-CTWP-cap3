# Classificação Automática de Grãos com Machine Learning

## Descrição do Projeto

Este projeto implementa um sistema de classificação automática de grãos de trigo utilizando técnicas de Machine Learning. O objetivo é automatizar o processo de classificação de três variedades de trigo (Kama, Rosa e Canadian) baseado em características físicas mensuráveis, substituindo o processo manual tradicionalmente realizado por especialistas em cooperativas agrícolas.

## Metodologia

O projeto segue a metodologia CRISP-DM (Cross-Industry Standard Process for Data Mining), abrangendo todas as etapas desde o entendimento do negócio até a implementação da solução.

## Dataset

**Fonte:** Seeds Dataset - UCI Machine Learning Repository
- **210 amostras** de grãos de trigo
- **3 classes:** Kama (1), Rosa (2), Canadian (3)
- **7 características físicas:**
  - Área do grão
  - Perímetro
  - Compacidade
  - Comprimento do núcleo
  - Largura do núcleo
  - Coeficiente de assimetria
  - Comprimento do sulco do núcleo

## Etapas do Projeto

### 1. Carregamento e Exploração dos Dados

O projeto inicia com o carregamento do dataset e análise exploratória básica:

```python
# Carregamento dos dados
df = pd.read_csv('seeds_dataset.txt', sep='\t', names=column_names)
print(f"Dataset: {df.shape[0]} amostras, {df.shape[1]} características")
```

**Resultado:** Dataset com 210 amostras balanceadas (70 por classe) e 7 características físicas.

### 2. Análise Estatística Descritiva

Cálculo de estatísticas descritivas para entender a distribuição dos dados:

| Característica | Média | Desvio Padrão | Mínimo | Máximo |
|----------------|-------|---------------|--------|--------|
| Área | 14.85 | 2.91 | 10.59 | 21.18 |
| Perímetro | 14.56 | 1.31 | 12.41 | 17.25 |
| Compacidade | 0.871 | 0.023 | 0.808 | 0.918 |

### 3. Visualização da Distribuição das Características

**Histogramas:** Mostram a distribuição de cada característica física dos grãos.

**Resultado observado:** A maioria das características apresenta distribuição aproximadamente normal, indicando boa qualidade dos dados para algoritmos de Machine Learning.

### 4. Análise de Outliers

**Boxplots:** Identificação de valores atípicos em cada característica.

**Conclusão:** Poucos outliers detectados, indicando consistência nas medições dos grãos.

### 5. Análise de Correlação

**Matriz de Correlação:** Identificação de relações lineares entre características.

**Correlações altas identificadas:**
- Área e Perímetro: 0.994
- Área e Comprimento do Núcleo: 0.993
- Perímetro e Comprimento do Núcleo: 0.991

**Implicação:** Algumas características são altamente correlacionadas, o que é esperado para medições físicas relacionadas.

### 6. Análise de Separabilidade das Classes

**Gráficos de Dispersão:** Visualização da separabilidade entre as três variedades de trigo.

**Observações:**
- Classes apresentam boa separabilidade visual
- Sobreposição mínima entre grupos
- Características como área e perímetro mostram distinção clara entre variedades

### 7. Pré-processamento dos Dados

**Divisão dos Dados:**
- Treinamento: 70% (147 amostras)
- Teste: 30% (63 amostras)
- Estratificação mantida para balanceamento das classes

**Normalização:**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Resultado:** Dados normalizados com média próxima de 0 e desvio padrão de 1.

### 8. Implementação dos Algoritmos de Classificação

**Cinco algoritmos implementados:**

1. **K-Nearest Neighbors (KNN)**
   - Algoritmo baseado em proximidade
   - Simples e interpretável

2. **Support Vector Machine (SVM)**
   - Classificador baseado em margens máximas
   - Eficaz para dados de alta dimensionalidade

3. **Random Forest**
   - Ensemble de árvores de decisão
   - Robusto e fornece importância das características

4. **Naive Bayes**
   - Classificador probabilístico
   - Rápido e eficiente

5. **Logistic Regression**
   - Regressão logística multinomial
   - Linear e interpretável

### 9. Avaliação Inicial dos Modelos

**Métricas utilizadas:**
- Acurácia
- Validação cruzada (5-fold)
- Matriz de confusão
- Precisão, Recall e F1-score

**Resultados iniciais:**
- Random Forest: 92.1% acurácia
- SVM: 90.5% acurácia
- Logistic Regression: 88.9% acurácia
- KNN: 87.3% acurácia
- Naive Bayes: 85.7% acurácia

### 10. Otimização de Hiperparâmetros

**Grid Search aplicado para:**

**Random Forest:**
```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
```

**SVM:**
```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
```

### 11. Resultados Finais Otimizados

**Performance após otimização:**

| Modelo | Acurácia | CV Média | Melhoria |
|--------|----------|----------|----------|
| Random Forest | 95.2% | 94.8% | +3.1% |
| SVM | 93.7% | 93.2% | +3.2% |
| Logistic Regression | 92.1% | 91.8% | +3.2% |
| KNN | 90.5% | 89.9% | +3.2% |
| Naive Bayes | 87.3% | 86.7% | +1.6% |

### 12. Análise do Melhor Modelo

**Random Forest - Melhor Performance:**
- **Acurácia:** 95.2%
- **Configuração otimizada:** n_estimators=100, max_depth=20

**Importância das Características:**
1. Área do grão: 23%
2. Perímetro: 19%
3. Comprimento do núcleo: 18%
4. Compacidade: 15%

### 13. Matriz de Confusão do Melhor Modelo

**Análise de Erros:**
- Total de erros: 3 de 63 amostras (4.8%)
- Maior confusão entre classes Kama e Rosa
- Classe Canadian apresentou melhor separabilidade

### 14. Análise de Correlações Finais

**Heatmap de Correlações:** Visualização das relações entre todas as características.

**Insights:**
- Características geométricas (área, perímetro) altamente correlacionadas
- Coeficiente de assimetria apresenta menor correlação com outras características
- Correlações altas não prejudicaram significativamente a performance dos modelos

## Resultados e Conclusões

### Performance Alcançada
- **Melhor modelo:** Random Forest com 95.2% de acurácia
- **Objetivo atingido:** Acurácia superior a 90%
- **Robustez:** Validação cruzada confirma consistência dos resultados

### Aplicabilidade Prática
- **Redução de tempo:** Classificação automática vs. processo manual
- **Redução de erros:** Minimização de erro humano
- **Escalabilidade:** Sistema pode processar grandes volumes de grãos
- **Custo-benefício:** Implementação viável para cooperativas

### Características Mais Relevantes
1. **Área do grão:** Principal fator discriminante
2. **Perímetro:** Segunda característica mais importante
3. **Comprimento do núcleo:** Terceira em importância

### Limitações Identificadas
- Dataset relativamente pequeno (210 amostras)
- Necessidade de validação com dados de diferentes origens
- Dependência da qualidade das medições dos equipamentos

### Recomendações para Implementação
- **Calibração regular** dos equipamentos de medição
- **Coleta de dados adicionais** para aumentar robustez
- **Treinamento de operadores** para uso do sistema
- **Monitoramento contínuo** da performance em produção

## Como Executar o Projeto

### Pré-requisitos
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly jupyter
```

### Execução
1. **Jupyter Notebook:**
   ```bash
   jupyter notebook Classificacao_Graos_ML.ipynb
   ```

2. **Google Colab:**
   - Acesse: https://colab.research.google.com/drive/1ExPftFMlgONNhaZB297uIAaE2jOSCI49
   - Execute as células sequencialmente

### Tempo de Execução
- **Total:** 5-10 minutos
- **Por seção:** 30-60 segundos

## Estrutura do Código

1. **Importações e Configuração**
2. **Carregamento de Dados**
3. **Análise Exploratória**
4. **Pré-processamento**
5. **Modelagem**
6. **Otimização**
7. **Avaliação**
8. **Interpretação de Resultados**

## Tecnologias Utilizadas

- **Python 3.8+**
- **Scikit-learn:** Algoritmos de Machine Learning
- **Pandas:** Manipulação de dados
- **NumPy:** Operações matemáticas
- **Matplotlib/Seaborn:** Visualização
- **Plotly:** Gráficos interativos

## Autor

**Richard Schmitz - RM567951**  
Projeto desenvolvido para a disciplina de Machine Learning - FIAP  
Novembro 2024