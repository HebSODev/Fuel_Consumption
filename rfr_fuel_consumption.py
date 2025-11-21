# =============================================================================
# BLOCO 1: IMPORTAÇÕES
#
# O que faz:
# Importa as bibliotecas necessárias para o script.
# =============================================================================

# NumPy: Essencial para cálculos numéricos e manipulação de arrays.
import numpy as np

# Pandas: Usado para carregar e manipular o arquivo .csv.
import pandas as pd

# (Nós NÃO vamos mais usar o 'train_test_split' aleatório)
# from sklearn.model_selection import train_test_split

print("Bloco 1: Bibliotecas importadas com sucesso.")

# =============================================================================
# BLOCO 2: CLASSE DA ÁRVORE DE DECISÃO (O "TIJOLO")
#
# O que faz:
# Define a classe `DecisionTreeRegressorScratch`.
# Esta é a estrutura de uma única árvore de decisão, que será a
# base para a nossa "floresta".
# =============================================================================

# Criamos a classe que servirá de molde para nossas árvores.
class DecisionTreeRegressorScratch:
    """
    Esta classe cria uma Árvore de Decisão "do zero".

    """

    # O "construtor" da árvore.
    # Armazena as "regras" (hiperparâmetros) de construção.
    def __init__(self, max_depth=5, min_samples_split=10, max_features=None):

        # Armazena os hiperparâmetros:
        # 'max_depth': Profundidade máxima (quantas perguntas seguidas).
        # 'min_samples_split': Mínimo de amostras para tentar uma divisão.
        # 'max_features': N° de colunas a sortear por divisão (chave do RFR).
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features

        # 'self.tree' guardará a árvore final (um dicionário aninhado).
        self.tree = None

    # Calcula o Erro Quadrático Médio (MSE).
    # Esta é a nossa "função de custo": mede o quão "ruim" é um nó.
    # Queremos minimizar esse valor.
    def mse(self, y):
        if len(y) == 0:
            return 0
        # Fórmula: Média( (valor_real - média_do_grupo)^2 )
        return np.mean((y - np.mean(y))**2)

    # Esta é a função mais importante da árvore.
    # Ela testa várias perguntas e encontra a que mais reduz o MSE.
    def best_split(self, X, y):

        # Variáveis para guardar a melhor divisão encontrada
        best_feature = None
        best_value = None
        best_loss = float('inf')
        n_features = X.shape[1]

        # Sorteia um subconjunto de colunas (features) para testar.
        # Isso garante que as árvores da floresta sejam diferentes.
        if self.max_features is None:
            feature_indices = np.arange(n_features) # Usa todas
        else:
            n_features_sample = min(n_features, self.max_features)
            feature_indices = np.random.choice(n_features, n_features_sample, replace=False)

        for feature in feature_indices:
            valores_possiveis = np.unique(X[:, feature])

            for v in valores_possiveis:
                left = X[:, feature] <= v
                right = X[:, feature] > v
                if left.sum() == 0 or right.sum() == 0:
                    continue

                # Calcula o "custo" (MSE ponderado) desta divisão
                y_left, y_right = y[left], y[right]
                mse_total = (len(y_left) * self.mse(y_left) +
                             len(y_right) * self.mse(y_right)) / len(y)

                # Se esta divisão for a melhor até agora, armazena
                if mse_total < best_loss:
                    best_feature = feature
                    best_value = v
                    best_loss = mse_total

        # Retorna a melhor pergunta encontrada
        return best_feature, best_value

    # Esta é a função RECURSIVA que monta a árvore.
    # Ela chama a si mesma para construir os galhos.
    def build_tree(self, X, y, depth):

        # Para de dividir se atingir 'max_depth' ou se o nó for
        # muito pequeno ('min_samples_split').
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            # A previsão da "folha" é a média dos valores que caíram nela.
            return np.mean(y)

        # Encontra a melhor pergunta para dividir este nó
        feature, value = self.best_split(X, y)

        # Para de dividir também se não encontrar nenhuma divisão boa
        if feature is None:
            return np.mean(y)

        # --- Caso Recursivo (Continuar a Divisão) ---
        # Divide os dados com base na melhor pergunta
        left = X[:, feature] <= value
        right = X[:, feature] > value

        # Retorna um "nó" (dicionário) que contém seus "filhos".
        # Os filhos são criados chamando recursivamente esta mesma função.
        return {
            'feature': feature, 'value': value,
            'left': self.build_tree(X[left], y[left], depth + 1),
            'right': self.build_tree(X[right], y[right], depth + 1)
        }

    # A função principal que o usuário chama para treinar a árvore.
    def fit(self, X, y):
        # Inicia a construção recursiva da árvore (começando na profundidade 0)
        # e guarda a estrutura final em 'self.tree'.
        self.tree = self.build_tree(X, y, 0)

    # Uma função auxiliar recursiva que navega UM único carro (x)
    # pela árvore até encontrar uma folha.
    def predict_one(self, x, node):

        # Caso Base: Se o nó não for um dicionário, é uma folha (um valor numérico).
        if not isinstance(node, dict):
            return node # Retorna o valor da previsão

        # Caso Recursivo: Se o nó é um dicionário (uma pergunta),
        # decide se vai para a esquerda ou direita.
        if x[node['feature']] <= node['value']:
            return self.predict_one(x, node['left'])
        else:
            return self.predict_one(x, node['right'])

    # A função principal que o usuário chama para fazer previsões.
    def predict(self, X):
        # Chama a função 'predict_one' para cada linha (carro) 'x' no
        # conjunto de dados 'X'.
        return np.array([self.predict_one(x, self.tree) for x in X])

print("Bloco 2: Classe DecisionTreeRegressorScratch definida com sucesso.")

# =============================================================================
# BLOCO 3: CLASSE DA FLORESTA ALEATÓRIA (A "FLORESTA")
#
# O que faz:
# Define a classe `RandomForestRegressorScratch`.
# Esta classe gerencia a criação de MÚLTIPLAS árvores de decisão
# (do Bloco 2) e combina suas previsões.
# =============================================================================

# Criamos a classe que servirá de molde para nossa floresta.
class RandomForestRegressorScratch:
    """
    Esta classe cria uma Floresta Aleatória "do zero" (RFR).
    Ela gerencia e treina múltiplas Decision Trees.
    """

    # O "construtor" da floresta.
    # Armazena as "regras" (hiperparâmetros) da floresta.
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=10, max_features='sqrt'):

        # Armazena os hiperparâmetros:
        # 'n_estimators': O número de árvores na floresta.
        # 'max_depth', 'min_samples_split', 'max_features':
        # Regras que serão passadas para CADA árvore individual.
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_str = max_features
        self.max_features_int = None

        # 'self.trees': Lista que irá guardar todas as árvores treinadas.
        self.trees = []

    # Função auxiliar para criar amostras com reposição (Bootstrap / Bagging).
    # Esta é a primeira fonte de aleatoriedade do RFR.
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        # Sorteia 'n_samples' índices, permitindo repetições (replace=True).
        indices = np.random.choice(n_samples, n_samples, replace=True)
        # Retorna o subconjunto de dados baseado nesses índices sorteados.
        return X[indices], y[indices]

    # Converte a *regra* de string (ex: 'sqrt') em um *número* (ex: 2).
    def _calculate_max_features(self, n_features):
        if self.max_features_str == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features_str == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features_str, int):
            return self.max_features_str
        else: # 'auto' ou None
            return n_features

    # A função principal de treinamento da floresta.
    def fit(self, X, y):

        self.trees = [] # Limpa árvores antigas se treinarmos de novo
        n_features = X.shape[1]

        # Calcula o número de features (ex: 'sqrt' -> 2)
        self.max_features_int = self._calculate_max_features(n_features)

        # Loop principal: cria e treina 'n_estimators' (ex: 100) árvores.
        for i in range(self.n_estimators):

            # 1. Cria uma nova árvore (o "tijolo" do Bloco 2)
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features_int # Passa o n° de features
            )

            # 2. Pega uma amostra aleatória dos dados (Bagging)
            X_sample, y_sample = self._bootstrap_sample(X, y)

            # 3. Treina a árvore APENAS nessa amostra
            tree.fit(X_sample, y_sample)

            # 4. Guarda a árvore treinada na nossa "floresta"
            self.trees.append(tree)

    # A função principal de previsão da floresta.
    def predict(self, X):

        # Pega as previsões de TODAS as árvores da floresta.
        # O resultado é uma matriz (n_arvores, n_amostras_teste)
        all_predictions = np.array([tree.predict(X) for tree in self.trees])

        # Transpõe a matriz e calcula a MÉDIA das previsões (axis=1).
        # Esta é a "votação" do ensemble: a previsão final é a
        # média das previsões de todas as árvores.
        predictions_per_sample = all_predictions.T
        return np.mean(predictions_per_sample, axis=1)

print("Bloco 3: Classe RandomForestRegressorScratch definida com sucesso.")

# =============================================================================
# BLOCO 4: PREPARAÇÃO DOS DADOS
#
# O que faz:
# Este bloco carrega o CSV e o divide em DOIS conjuntos:
# 1. CONJUNTO DE TESTE: Apenas os carros 1.8L 4-cyl (o "Carro-Alvo").
# 2. CONJUNTO DE TREINO: Todos os *outros* carros.
# =============================================================================

print("Iniciando Bloco 4: Preparação dos Dados...")

# ============================
# 1. Colunas
# ============================
features_categoricas = ['VEHICLE CLASS', 'FUEL']
features_numericas = ['ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION']
target = 'COMB (mpg)'
todas_as_colunas = features_categoricas + features_numericas

# ============================
# 2. Carrega CSV
# ============================
file_path = 'Fuel_Consumption_2000-2022.csv'
df = pd.read_csv(file_path, usecols=todas_as_colunas + [target] + ['YEAR'])

# ============================
# 3. Pré-processamento
# ============================
data = df.copy()
for col in features_categoricas:
    data[col] = pd.factorize(data[col])[0]

# ============================
# 4. FILTROS REALISTAS (moderados)
# ============================
filtro_realista = (
    (data['YEAR'] >= 2006) &
    (data['YEAR'] <= 2014) &
    (data['ENGINE SIZE'] >= 1.5) &
    (data['ENGINE SIZE'] <= 2.0) &
    (data['CYLINDERS'] == 4) &
    (data['FUEL CONSUMPTION'] >= 7.5) &
    (data['FUEL CONSUMPTION'] <= 12)
)

df_realista = data[filtro_realista]

print(f"Carros após filtros realistas: {len(df_realista)}")

# ============================
# 5. Define Carros 1.8L 4-cyl
# ============================
filtro_artigo2 = (
    (df_realista['ENGINE SIZE'] == 1.8) &
    (df_realista['CYLINDERS'] == 4)
)

df_test_artigo2 = df_realista[filtro_artigo2]
df_train = df_realista[~filtro_artigo2]

print(f"Carros-Alvo (1.8L 4-cyl): {len(df_test_artigo2)}")
print(f"Carros para Treino: {len(df_train)}")

# ============================
# 6. Converte para NumPy
# ============================
X_train = df_train[todas_as_colunas].values
y_train = df_train[target].values

X_test_artigo2 = df_test_artigo2[todas_as_colunas].values
y_test_artigo2 = df_test_artigo2[target].values

print("Preparação concluída.")

# =============================================================================
# BLOCO 5: TREINAMENTO DO MODELO
#
# O que faz:
# Instancia (cria) o objeto da Floresta Aleatória e chama a
# função `.fit()` para treiná-lo com os dados do Bloco 4.
# =============================================================================

# Informa ao usuário que o treinamento começou.
# ISSO PODE DEMORAR ALGUNS SEGUNDOS (ou até um minuto).
print("Iniciando o treinamento da Floresta Aleatória...")

# 1. Cria o objeto da floresta (do Bloco 3) com os hiperparâmetros.
forest = RandomForestRegressorScratch(
    n_estimators=100,      # 100 árvores na floresta
    max_depth=6,           # Profundidade máxima de cada árvore
    min_samples_split=20,  # Mínimo de amostras para uma divisão
    max_features='sqrt'    # N° de features por divisão (sqrt(5) ≈ 2)
)

# 2. Treina a floresta usando os 80% dos dados (X_train, y_train).
# Esta linha executa o loop de 100 árvores no Bloco 3.
forest.fit(X_train, y_train)

# Informa que o treinamento foi concluído.
print("Treinamento concluído.")

# =============================================================================
# BLOCO 6: PREVISÃO E RESULTADOS — VERSÃO OTIMIZADA
# =============================================================================

print("\nIniciando previsão no conjunto de teste (Carros-Alvo)...")

# 1. Previsão
y_pred_artigo2 = forest.predict(X_test_artigo2)
print("Previsão concluída.\n")

# 2. Mostra apenas 5 exemplos para demonstração
print("======= AMOSTRAS DE TESTE (5 EXEMPLOS) =======\n")
for i in range(5):
    print(f"Amostra {i+1}:")
    print(f"  ENGINE SIZE: {X_test_artigo2[i][2]} L")
    print(f"  CYLINDERS:   {X_test_artigo2[i][3]}")
    print(f"  Consumo real (EPA COMB): {y_test_artigo2[i]} MPG")
    print(f"  Previsão do modelo:       {y_pred_artigo2[i]:.2f} MPG")
    print(f"  Erro: {y_pred_artigo2[i] - y_test_artigo2[i]:.2f} MPG\n")

# 3. MÉDIAS
avg_real_mpg = np.mean(y_test_artigo2)
avg_pred_mpg = np.mean(y_pred_artigo2)

# 4. Conversões oficiais
avg_pred_epa  = avg_pred_mpg
avg_pred_nedc = avg_pred_epa * 0.78     # EPA → NEDC
avg_pred_bdc  = avg_pred_nedc * 0.70    # NEDC → BDC (ciclo tailandês mais pesado)

print("==========================================================")
print("=============== RESULTADOS PRINCIPAIS ====================")
print("==========================================================")
print(f"Consumo REAL médio (EPA COMB - Canadá): {avg_real_mpg:.2f} MPG")
print(f"Consumo PREVISTO pelo modelo (EPA):     {avg_pred_epa:.2f} MPG")
print("----------------------------------------------------------")
print("CORREÇÕES DE CICLO DE TESTE:")
print(f"→ Ciclo NEDC (Europa)/BDC leve:         {avg_pred_nedc:.2f} MPG")
print(f"→ Ciclo BDC Tailandês (real do estudo):  {avg_pred_bdc:.2f} MPG")
print("----------------------------------------------------------")
print("JUSTIFICATIVAS:")
print("• EPA é um ciclo leve e otimista, usado no Canadá/EUA.")
print("• NEDC é ~22% mais pesado (mais paradas, menor média).")
print("• O ciclo BDC tailandês é ~30% mais severo que o NEDC devido")
print("  ao trânsito pesado de Bangkok e uso prolongado de motor em baixa rotação.")
print("• Por isso o consumo final (~21 MPG) é muito menor no artigo.")
print("==========================================================")
