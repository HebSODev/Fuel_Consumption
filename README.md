# 🚗 Fuel Consumption Prediction

Projeto de Machine Learning para previsão de consumo de combustível de veículos utilizando diferentes algoritmos de regressão, incluindo Random Forest e Gradient Boosting.

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Objetivo](#objetivo)
- [Dataset](#dataset)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Instalação](#instalação)
- [Como Rodar](#como-rodar)
- [Modelos Implementados](#modelos-implementados)
- [Resultados](#resultados)
- [Conversões de Ciclos de Teste](#conversões-de-ciclos-de-teste)

## 🎯 Sobre o Projeto

Este projeto analisa dados de consumo de combustível de veículos no Canadá (2000-2022) e implementa modelos de Machine Learning para prever o consumo de combustível. O foco principal está em veículos com motor 1.8L de 4 cilindros, com análise comparativa entre diferentes ciclos de teste (EPA, NEDC, BDC).

## 🚀 Objetivo

- **Prever o consumo de combustível** (em MPG) de veículos com base em características técnicas
- **Comparar diferentes algoritmos** de Machine Learning (Random Forest, Gradient Boosting, XGBoost, LightGBM)
- **Converter previsões** entre diferentes ciclos de teste internacionais
- **Implementar Random Forest do zero** para fins educacionais e compreensão do algoritmo

## 📊 Dataset

O dataset utilizado é o **Fuel Consumption Ratings** (2000-2022) do governo canadense, contendo:

- **Período**: 2000 a 2022
- **Features principais**:
  - `ENGINE SIZE`: Tamanho do motor (L)
  - `CYLINDERS`: Número de cilindros
  - `FUEL CONSUMPTION`: Consumo de combustível (L/100km)
  - `VEHICLE CLASS`: Classe do veículo
  - `FUEL`: Tipo de combustível
- **Target**: `COMB (mpg)` - Consumo combinado em milhas por galão

**Filtros aplicados**:
- Anos: 2006-2014
- Motor: 1.5L - 2.0L
- Cilindros: 4
- Consumo: 7.5 - 12 L/100km

## 📁 Estrutura do Projeto

```
Fuel_Consumption/
│
├── Fuel_Consumption_2000-2022.csv          # Dataset principal
├── requirements.txt                         # Dependências do projeto
│
├── gradient_boosting_fuel_consumption.py   # Modelos prontos (scikit-learn, XGBoost, LightGBM)
├── rfr_fuel_consumption.py                 # Random Forest implementado do zero
│
├── output_visualizations/                   # Gráficos e visualizações geradas
│   ├── feature_importance.png              # Importância das features
│   ├── model_comparison_rmse.png           # Comparação de RMSE entre modelos
│   ├── real_vs_predicted.png               # Scatter plot real vs previsto
│   └── error_distribution.png              # Distribuição dos erros
│
└── slide/                                   # Apresentações e documentação adicional
```

## 🛠️ Tecnologias Utilizadas

### Linguagem
- **Python 3.8+**

### Bibliotecas Principais
- **NumPy** - Operações numéricas e arrays
- **Pandas** - Manipulação de dados
- **Scikit-learn** - Modelos de ML e métricas
- **XGBoost** - Gradient Boosting otimizado
- **LightGBM** - Gradient Boosting eficiente
- **Matplotlib** - Visualização de dados
- **Seaborn** - Visualizações estatísticas

## 💻 Instalação

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passo a Passo

1. **Clone o repositório**:
```powershell
git clone https://github.com/HebSODev/Fuel_Consumption.git
cd Fuel_Consumption
```

2. **Crie um ambiente virtual (recomendado)**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Instale as dependências**:
```powershell
pip install -r requirements.txt
```

## 🚀 Como Rodar

### Opção 1: Modelos com Bibliotecas Prontas (Recomendado)

Este script treina e compara 4 modelos diferentes:

```powershell
python gradient_boosting_fuel_consumption.py
```

**Saída esperada**:
- Comparação de métricas (RMSE, MAE, R²)
- Melhor modelo identificado
- Previsões no conjunto de teste
- Gráficos salvos em `output_visualizations/`

### Opção 2: Random Forest Implementado do Zero

Este script implementa o Random Forest manualmente para fins educacionais:

```powershell
python rfr_fuel_consumption.py
```

**Saída esperada**:
- Demonstração do funcionamento interno do algoritmo
- Previsões passo a passo
- Análise de 5 amostras de teste

### Visualizações Geradas

Após executar os scripts, os gráficos serão salvos em `output_visualizations/`:

1. **feature_importance.png** - Mostra quais features mais influenciam o modelo
2. **model_comparison_rmse.png** - Compara o erro (RMSE) entre os modelos
3. **real_vs_predicted.png** - Scatter plot comparando valores reais vs previstos
4. **error_distribution.png** - Histograma da distribuição dos erros de previsão

## 🤖 Modelos Implementados

### 1. Random Forest (Scikit-learn)
- **n_estimators**: 200 árvores
- **max_depth**: 10
- **max_features**: 'sqrt'

### 2. Gradient Boosting (Scikit-learn)
- **n_estimators**: 200
- **learning_rate**: 0.1
- **max_depth**: 5

### 3. XGBoost
- **n_estimators**: 200
- **learning_rate**: 0.1
- **max_depth**: 6
- **subsample**: 0.8

### 4. LightGBM
- **n_estimators**: 200
- **learning_rate**: 0.1
- **max_depth**: 6
- **num_leaves**: 31

### 5. Random Forest do Zero
Implementação manual do Random Forest com:
- **n_estimators**: 100
- **max_depth**: 6
- **max_features**: 'sqrt'
- Bootstrap sampling (Bagging)
- Feature randomization

## 📈 Resultados

### Métricas Típicas (Conjunto de Validação)

| Modelo | RMSE (MPG) | MAE (MPG) | R² |
|--------|------------|-----------|-----|
| LightGBM | ~1.5-2.0 | ~1.2-1.5 | ~0.85-0.90 |
| XGBoost | ~1.5-2.0 | ~1.2-1.5 | ~0.85-0.90 |
| Gradient Boosting | ~1.8-2.2 | ~1.4-1.7 | ~0.82-0.88 |
| Random Forest | ~2.0-2.5 | ~1.5-1.8 | ~0.80-0.85 |

*Os valores exatos podem variar dependendo dos dados e da aleatoriedade do treinamento.*

### Features Mais Importantes
1. **FUEL CONSUMPTION** - Maior impacto
2. **ENGINE SIZE** - Influência significativa
3. **CYLINDERS** - Importância moderada
4. **VEHICLE CLASS** - Influência menor
5. **FUEL** - Menor impacto

## 🌍 Conversões de Ciclos de Teste

O projeto inclui conversões entre diferentes padrões de teste de combustível:

### Ciclos de Teste Explicados

- **EPA (Environmental Protection Agency)**
  - Usado no Canadá e EUA
  - Ciclo mais otimista (condições ideais)
  - Base para as medições do dataset

- **NEDC (New European Driving Cycle)**
  - Usado na Europa
  - ~22% mais severo que o EPA
  - Conversão: `NEDC = EPA × 0.78`

- **BDC (Bangkok Driving Cycle)**
  - Usado na Tailândia
  - ~30% mais severo que o NEDC
  - Condições de tráfego intenso
  - Conversão: `BDC = NEDC × 0.70`

### Exemplo de Conversão

```
Previsão EPA:  30.00 MPG
↓ (× 0.78)
Ciclo NEDC:    23.40 MPG
↓ (× 0.70)
Ciclo BDC:     16.38 MPG
```

## 📝 Notas Adicionais

### Por que dois scripts diferentes?

1. **gradient_boosting_fuel_consumption.py**
   - Usa bibliotecas prontas e otimizadas
   - Melhor performance e precisão
   - Ideal para uso em produção
   - Compara múltiplos algoritmos

2. **rfr_fuel_consumption.py**
   - Implementação educacional
   - Demonstra o funcionamento interno do Random Forest
   - Útil para aprendizado e compreensão do algoritmo
   - Código comentado em detalhes

### Limitações

- Os modelos foram treinados com dados de 2006-2014
- Focado em veículos com motor 1.5L-2.0L de 4 cilindros
- As conversões de ciclos são aproximações baseadas em estudos
- O desempenho pode variar com dados de veículos mais recentes

## 📞 Contato

- **Autores**: HebSODev, Thales Albino
- **Repositórios**:[](https://github.com/Thales-P), [https://github.com/HebSODev/Fuel_Consumption](https://github.com/HebSODev/Fuel_Consumption)

---

⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!
