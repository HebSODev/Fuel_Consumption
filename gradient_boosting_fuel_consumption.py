# =============================================================================
# MODELO DE GRADIENT BOOSTING PARA PREVISÃO DE CONSUMO DE COMBUSTÍVEL
# Usando bibliotecas prontas: scikit-learn, XGBoost, LightGBM
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Criar diretório para salvar as visualizações
OUTPUT_DIR = Path('output_visualizations')
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Diretório de saída: {OUTPUT_DIR.absolute()}")

print("="*70)
print("ANÁLISE DE CONSUMO DE COMBUSTÍVEL - MÚLTIPLOS MODELOS ML")
print("="*70)

# =============================================================================
# BLOCO 1: CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# =============================================================================

print("\n[1] Carregando e preparando os dados...")

# Definição das colunas
features_categoricas = ['VEHICLE CLASS', 'FUEL']
features_numericas = ['ENGINE SIZE', 'CYLINDERS', 'FUEL CONSUMPTION']
target = 'COMB (mpg)'
todas_as_colunas = features_categoricas + features_numericas

# Carrega o CSV
file_path = 'Fuel_Consumption_2000-2022.csv'
df = pd.read_csv(file_path, usecols=todas_as_colunas + [target] + ['YEAR'])

print(f"   • Dataset original: {len(df)} registros")

# Pré-processamento
data = df.copy()
for col in features_categoricas:
    data[col] = pd.factorize(data[col])[0]

# Aplicação de filtros realistas
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
print(f"   • Após filtros realistas: {len(df_realista)} registros")

# Separação: Carros-Alvo (1.8L 4-cyl) vs Treino
filtro_artigo = (
    (df_realista['ENGINE SIZE'] == 1.8) &
    (df_realista['CYLINDERS'] == 4)
)

df_test_artigo = df_realista[filtro_artigo]
df_train = df_realista[~filtro_artigo]

print(f"   • Carros-Alvo (1.8L 4-cyl): {len(df_test_artigo)} registros")
print(f"   • Carros para Treino: {len(df_train)} registros")

# Conversão para arrays NumPy
X_train = df_train[todas_as_colunas].values
y_train = df_train[target].values
X_test_artigo = df_test_artigo[todas_as_colunas].values
y_test_artigo = df_test_artigo[target].values

# Divisão adicional para validação durante o treino
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"   • Train set: {len(X_train_split)} | Validation set: {len(X_val_split)}")

# =============================================================================
# BLOCO 2: NORMALIZAÇÃO DOS DADOS
# =============================================================================

print("\n[2] Normalizando features numéricas...")

scaler = StandardScaler()
X_train_split_scaled = scaler.fit_transform(X_train_split)
X_val_split_scaled = scaler.transform(X_val_split)
X_test_scaled = scaler.transform(X_test_artigo)

# =============================================================================
# BLOCO 3: TREINAMENTO DE MÚLTIPLOS MODELOS
# =============================================================================

print("\n[3] Treinando múltiplos modelos de Machine Learning...")
print("-" * 70)

modelos = {}

# --- 3.1 Random Forest (Scikit-learn) ---
print("\n   [3.1] Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_split, y_train_split)
modelos['Random Forest'] = rf_model
print("   ✓ Random Forest treinado")

# --- 3.2 Gradient Boosting (Scikit-learn) ---
print("\n   [3.2] Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
gb_model.fit(X_train_split, y_train_split)
modelos['Gradient Boosting'] = gb_model
print("   ✓ Gradient Boosting treinado")

# --- 3.3 XGBoost ---
print("\n   [3.3] XGBoost Regressor...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_split, y_train_split)
modelos['XGBoost'] = xgb_model
print("   ✓ XGBoost treinado")

# --- 3.4 LightGBM ---
print("\n   [3.4] LightGBM Regressor...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    num_leaves=31,
    min_child_samples=10,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train_split, y_train_split)
modelos['LightGBM'] = lgb_model
print("   ✓ LightGBM treinado")

# =============================================================================
# BLOCO 4: AVALIAÇÃO DOS MODELOS NO CONJUNTO DE VALIDAÇÃO
# =============================================================================

print("\n" + "="*70)
print("[4] AVALIAÇÃO DOS MODELOS (Conjunto de Validação)")
print("="*70)

resultados = []

for nome, modelo in modelos.items():
    # Previsões no conjunto de validação
    y_pred_val = modelo.predict(X_val_split)
    
    # Métricas
    mse = mean_squared_error(y_val_split, y_pred_val)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val_split, y_pred_val)
    r2 = r2_score(y_val_split, y_pred_val)
    
    resultados.append({
        'Modelo': nome,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    })
    
    print(f"\n{nome}:")
    print(f"   RMSE: {rmse:.4f} MPG")
    print(f"   MAE:  {mae:.4f} MPG")
    print(f"   R²:   {r2:.4f}")

# Tabela comparativa
print("\n" + "-"*70)
print("RESUMO COMPARATIVO:")
print("-"*70)
df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values('RMSE')
print(df_resultados.to_string(index=False))

# Melhor modelo
melhor_modelo_nome = df_resultados.iloc[0]['Modelo']
melhor_modelo = modelos[melhor_modelo_nome]
print(f"\n🏆 Melhor modelo: {melhor_modelo_nome}")

# =============================================================================
# BLOCO 5: PREVISÃO NO CONJUNTO DE TESTE (CARROS-ALVO 1.8L 4-CYL)
# =============================================================================

print("\n" + "="*70)
print("[5] PREVISÃO NO CONJUNTO DE TESTE (Carros-Alvo 1.8L 4-cyl)")
print("="*70)

# Previsões de todos os modelos no conjunto de teste
previsoes_teste = {}
for nome, modelo in modelos.items():
    previsoes_teste[nome] = modelo.predict(X_test_artigo)

# Exibir 5 exemplos
print("\n📊 AMOSTRAS DE TESTE (5 exemplos):\n")
for i in range(min(5, len(X_test_artigo))):
    print(f"Amostra {i+1}:")
    print(f"  ENGINE SIZE: {X_test_artigo[i][2]} L")
    print(f"  CYLINDERS:   {X_test_artigo[i][3]}")
    print(f"  Consumo REAL (EPA): {y_test_artigo[i]:.2f} MPG")
    print(f"  Previsões dos modelos:")
    for nome in modelos.keys():
        print(f"    • {nome:20s}: {previsoes_teste[nome][i]:.2f} MPG")
    print()

# =============================================================================
# BLOCO 6: ANÁLISE FINAL COM CONVERSÕES DE CICLO
# =============================================================================

print("="*70)
print("[6] RESULTADOS FINAIS - CONVERSÃO DE CICLOS DE TESTE")
print("="*70)

avg_real_mpg = np.mean(y_test_artigo)
print(f"\n📌 Consumo REAL médio (EPA COMB - Canadá): {avg_real_mpg:.2f} MPG\n")

print("PREVISÕES POR MODELO:")
print("-"*70)

for nome, modelo in modelos.items():
    avg_pred_epa = np.mean(previsoes_teste[nome])
    avg_pred_nedc = avg_pred_epa * 0.78  # EPA → NEDC
    avg_pred_bdc = avg_pred_nedc * 0.70  # NEDC → BDC (Tailândia)
    
    # Métricas no teste
    rmse_test = np.sqrt(mean_squared_error(y_test_artigo, previsoes_teste[nome]))
    mae_test = mean_absolute_error(y_test_artigo, previsoes_teste[nome])
    
    print(f"\n{nome}:")
    print(f"  • Previsão EPA (média):  {avg_pred_epa:.2f} MPG")
    print(f"  • Ciclo NEDC (Europa):   {avg_pred_nedc:.2f} MPG")
    print(f"  • Ciclo BDC (Tailândia): {avg_pred_bdc:.2f} MPG")
    print(f"  • RMSE no teste:         {rmse_test:.4f} MPG")
    print(f"  • MAE no teste:          {mae_test:.4f} MPG")

# =============================================================================
# BLOCO 7: ANÁLISE DE IMPORTÂNCIA DE FEATURES (MELHOR MODELO)
# =============================================================================

print("\n" + "="*70)
print(f"[7] IMPORTÂNCIA DAS FEATURES ({melhor_modelo_nome})")
print("="*70)

if hasattr(melhor_modelo, 'feature_importances_'):
    importancias = melhor_modelo.feature_importances_
    nomes_features = todas_as_colunas
    
    df_importance = pd.DataFrame({
        'Feature': nomes_features,
        'Importância': importancias
    }).sort_values('Importância', ascending=False)
    
    print("\n" + df_importance.to_string(index=False))
    
    # Visualização (salvando como arquivo)
    plt.figure(figsize=(10, 6))
    plt.barh(df_importance['Feature'], df_importance['Importância'])
    plt.xlabel('Importância')
    plt.title(f'Importância das Features - {melhor_modelo_nome}')
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'feature_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Gráfico salvo: {output_path}")
    plt.close()

# =============================================================================
# BLOCO 8: VISUALIZAÇÃO COMPARATIVA
# =============================================================================

print("\n" + "="*70)
print("[8] GERANDO VISUALIZAÇÕES COMPARATIVAS")
print("="*70)

# Gráfico 1: Comparação de RMSE
plt.figure(figsize=(10, 6))
plt.bar(df_resultados['Modelo'], df_resultados['RMSE'], color='steelblue')
plt.xlabel('Modelo', fontsize=12)
plt.ylabel('RMSE (MPG)', fontsize=12)
plt.title('Comparação de RMSE entre Modelos', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
output_path = OUTPUT_DIR / 'model_comparison_rmse.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"📊 Gráfico salvo: {output_path}")
plt.close()

# Gráfico 2: Real vs Previsto (Melhor Modelo)
plt.figure(figsize=(10, 8))
y_pred_melhor = previsoes_teste[melhor_modelo_nome]

plt.scatter(y_test_artigo, y_pred_melhor, alpha=0.6, s=50)
plt.plot([y_test_artigo.min(), y_test_artigo.max()], 
         [y_test_artigo.min(), y_test_artigo.max()], 
         'r--', lw=2, label='Previsão Perfeita')
plt.xlabel('Consumo Real (MPG)', fontsize=12)
plt.ylabel('Consumo Previsto (MPG)', fontsize=12)
plt.title(f'Real vs Previsto - {melhor_modelo_nome}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
output_path = OUTPUT_DIR / 'real_vs_predicted.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"📊 Gráfico salvo: {output_path}")
plt.close()

# Gráfico 3: Distribuição dos Erros
plt.figure(figsize=(10, 6))
erros = y_pred_melhor - y_test_artigo
plt.hist(erros, bins=30, color='coral', edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erro Zero')
plt.xlabel('Erro de Previsão (MPG)', fontsize=12)
plt.ylabel('Frequência', fontsize=12)
plt.title(f'Distribuição dos Erros - {melhor_modelo_nome}', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
output_path = OUTPUT_DIR / 'error_distribution.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"📊 Gráfico salvo: {output_path}")
plt.close()

print("\n" + "="*70)
print("✅ ANÁLISE COMPLETA!")
print("="*70)
print(f"\nArquivos gerados em '{OUTPUT_DIR}':")
print("  • feature_importance.png")
print("  • model_comparison_rmse.png")
print("  • real_vs_predicted.png")
print("  • error_distribution.png")
print("\n" + "="*70)
