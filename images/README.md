# Imagens e Gráficos do Projeto

## Como Adicionar Gráficos

### 1. No Google Colab, salve cada gráfico:

```python
# Para gráficos Matplotlib/Seaborn
plt.savefig('nome_do_grafico.png', dpi=300, bbox_inches='tight')

# Para gráficos Plotly
fig.write_image('nome_do_grafico.png', width=800, height=600)
```

### 2. Faça download dos arquivos PNG

### 3. Adicione nesta pasta: `/images/graficos/`

## Gráficos Necessários:

- `distribuicao_caracteristicas.png` - Histogramas das 7 características
- `boxplots_outliers.png` - Boxplots para detecção de outliers  
- `matriz_correlacao.png` - Heatmap de correlações
- `dispersao_classes.png` - Gráficos de dispersão entre classes
- `performance_modelos.png` - Comparação de acurácia dos modelos
- `matriz_confusao_melhor.png` - Matriz de confusão do melhor modelo
- `importancia_caracteristicas.png` - Importância das features (Random Forest)
- `comparacao_otimizacao.png` - Antes vs depois da otimização

## Uso no README:

```markdown
![Distribuição das Características](images/graficos/distribuicao_caracteristicas.png)
```