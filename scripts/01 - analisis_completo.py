# %% [markdown]
# # Análisis Completo de Datos de Bancarrota
# 
# En este script realizaremos un análisis completo de los datos de bancarrota, incluyendo:
# - Análisis exploratorio de datos
# - Visualización de distribuciones
# - Análisis de correlaciones
# - Análisis de componentes principales
# - Importancia de variables

# %% [code]
# Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]

# Configurar directorio de visualizaciones
import os
os.makedirs('../visualizaciones', exist_ok=True)

class AnalisisBancarrota:
    def __init__(self):
        # Criar diretório para visualizações
        print("1. CARGANDO DATOS")
        print("-" * 50)
        self.train_data = pd.read_csv('../archivos/train_data.csv')
        self.test_data = pd.read_csv('../archivos/test_data.csv')
        
        # Separar features numéricas
        self.numeric_features = self.train_data.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_features.remove('ID')
        self.numeric_features.remove('Bankruptcy')
    
    def analisis_exploratorio(self):
        """Análise exploratória básica dos dados"""
        print("\n2. ANÁLISIS EXPLORATORIO BÁSICO")
        print("-" * 50)
        
        # Informações básicas
        print("\nInformación básica del dataset:")
        print(self.train_data.info())
        
        # Estatísticas descritivas
        print("\nEstadísticas descriptivas:")
        print(self.train_data.describe())
        
        # Valores nulos
        print("\nValores nulos por columna:")
        print(self.train_data.isnull().sum())
        
        # Distribuição da target
        print("\nDistribución de la variable Bankruptcy:")
        print(self.train_data['Bankruptcy'].value_counts(normalize=True))
        
        # 1. Distribuição da target
        plt.figure(figsize=(12, 8))
        sns.countplot(x='Bankruptcy', data=self.train_data)
        plt.title('Distribución de Bancarrota')
        plt.savefig('../visualizaciones/01_distribucion_bancarrota.png')
        plt.close()
        
        # 2. Boxplot de lucro operacional
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Bankruptcy', y='Operating.Profit.Rate', data=self.train_data)
        plt.title('Tasa de Beneficio Operativo por Estado')
        plt.savefig('../visualizaciones/02_beneficio_operativo.png')
        plt.close()
        
        # 3. Histograma de passivo corrente
        plt.figure(figsize=(12, 8))
        sns.histplot(data=self.train_data, x='Current.Liability.to.Assets', 
                    hue='Bankruptcy', kde=True)
        plt.title('Distribución de Pasivo Corriente/Activos')
        plt.savefig('../visualizaciones/03_pasivo_corriente.png')
        plt.close()
        
        # 4. Matriz de correlação inicial
        plt.figure(figsize=(16, 12))
        corr_matrix = self.train_data.iloc[:, 2:12].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlación (Primeras 10 Variables)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('../visualizaciones/04_correlacion_inicial.png')
        plt.close()
        
        # 5. Boxplot de ratio de dívida
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Bankruptcy', y='Debt.ratio..', data=self.train_data)
        plt.title('Ratio de Deuda por Estado')
        plt.savefig('../visualizaciones/05_ratio_deuda.png')
        plt.close()
        
        # 6. Histograma de ROA
        plt.figure(figsize=(12, 8))
        sns.histplot(data=self.train_data, x='ROA.B..before.interest.and.depreciation.after.tax', 
                    hue='Bankruptcy', kde=True)
        plt.title('Distribución de ROA')
        plt.savefig('../visualizaciones/06_distribucion_roa.png')
        plt.close()
    
    def analisis_avanzado(self):
        """Análise avançada com foco em correlações e comparações"""
        print("\n3. ANÁLISIS AVANZADO")
        print("-" * 50)
        
        # Análise de correlação
        correlations = self.train_data.corr()['Bankruptcy'].sort_values(ascending=False)
        print("\nTop 10 variables más positivamente correlacionadas con Bancarrota:")
        print(correlations.head(11))
        print("\nTop 10 variables más negativamente correlacionadas con Bancarrota:")
        print(correlations.tail(10))
        
        # Separar empresas
        failed = self.train_data[self.train_data['Bankruptcy'] == 1]
        healthy = self.train_data[self.train_data['Bankruptcy'] == 0]
        
        # 7. Top correlações
        plt.figure(figsize=(12, 10))
        top_corr = pd.concat([correlations.head(6), correlations.tail(5)])
        sns.barplot(x=top_corr.values, y=top_corr.index)
        plt.title('Top Correlaciones con Bancarrota')
        plt.xlabel('Coeficiente de Correlación')
        plt.tight_layout()
        plt.savefig('../visualizaciones/07_top_correlaciones.png')
        plt.close()
        
        # 8. Comparação de distribuições
        important_vars = correlations.head(6).index.tolist()[1:]
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Bankruptcy', y=important_vars[0], data=self.train_data)
        plt.title('Comparación de Distribuciones - Variable Más Correlacionada')
        plt.savefig('../visualizaciones/08_distribucion_top_variable.png')
        plt.close()
        
        # 9. Scatter plot
        plt.figure(figsize=(12, 8))
        var1, var2 = important_vars[0], important_vars[1]
        plt.scatter(self.train_data[var1], self.train_data[var2], 
                   c=self.train_data['Bankruptcy'], cmap='coolwarm', alpha=0.6)
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.title('Relación entre las Dos Variables Más Correlacionadas')
        plt.savefig('../visualizaciones/09_scatter_correlaciones.png')
        plt.close()
        
        # 10. Densidade
        plt.figure(figsize=(12, 8))
        sns.kdeplot(data=self.train_data, x='Operating.Profit.Rate', 
                   hue='Bankruptcy', common_norm=False)
        plt.title('Densidad de Tasa de Beneficio Operativo por Estado')
        plt.savefig('../visualizaciones/10_densidad_beneficio.png')
        plt.close()
        
        # 11 e 12. Violin plots
        plt.figure(figsize=(12, 8))
        sns.violinplot(x='Bankruptcy', y=important_vars[0], data=self.train_data)
        plt.title(f'Distribución de {important_vars[0]}')
        plt.savefig('../visualizaciones/11_violin_var1.png')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        sns.violinplot(x='Bankruptcy', y=important_vars[1], data=self.train_data)
        plt.title(f'Distribución de {important_vars[1]}')
        plt.savefig('../visualizaciones/12_violin_var2.png')
        plt.close()
        
        # Análise estatística
        print("\nAnálisis Estadístico de Diferencias entre Grupos:")
        for var in important_vars[:5]:
            stat, p_value = stats.mannwhitneyu(failed[var], healthy[var], 
                                             alternative='two-sided')
            print(f"\n{var}:")
            print(f"Mediana (Quiebra): {failed[var].median():.4f}")
            print(f"Mediana (Solvente): {healthy[var].median():.4f}")
            print(f"p-value: {p_value:.4e}")
    
    def analisis_completo(self):
        """Análise completa com PCA e testes estatísticos"""
        print("\n4. ANÁLISIS COMPLETO")
        print("-" * 50)
        
        # Preparar dados
        X = self.train_data[self.numeric_features]
        y = self.train_data['Bankruptcy']
        
        # Normalizar dados
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Análise PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print("\nVarianza explicada por componente:")
        for i, var in enumerate(explained_variance[:10]):
            print(f"PC{i+1}: {var:.4f} ({cumulative_variance[i]:.4f} acumulado)")
        
        # Importância das features
        mi_scores = mutual_info_classif(X_scaled, y)
        feature_importance = pd.DataFrame({
            'Variable': self.numeric_features,
            'Puntuación_MI': mi_scores
        }).sort_values('Puntuación_MI', ascending=False)
        
        print("\nTop 10 variables más importantes (Mutual Information):")
        print(feature_importance.head(10))
        
        # 13. PCA
        plt.figure(figsize=(12, 8))
        plt.plot(range(1, len(explained_variance) + 1), 
                cumulative_variance, 'bo-')
        plt.xlabel('Número de Componentes')
        plt.ylabel('Varianza Explicada Acumulada')
        plt.title('Análisis de Componentes Principales')
        plt.grid(True)
        plt.savefig('../visualizaciones/13_pca_varianza.png')
        plt.close()
        
        # 14. Importância das features
        plt.figure(figsize=(12, 10))
        sns.barplot(data=feature_importance.head(10), 
                   x='Puntuación_MI', y='Variable')
        plt.title('Top 10 Variables Más Importantes')
        plt.tight_layout()
        plt.savefig('../visualizaciones/14_importancia_variables.png')
        plt.close()
        
        # 15. Distribuições condicionais
        plt.figure(figsize=(12, 8))
        top_feature = feature_importance['Variable'].iloc[0]
        sns.kdeplot(data=self.train_data, x=top_feature, 
                   hue='Bankruptcy', common_norm=False)
        plt.title(f'Distribución de {top_feature} por Estado')
        plt.savefig('../visualizaciones/15_distribucion_top_mi.png')
        plt.close()
        
        # 16. Matriz de correlação das top features
        plt.figure(figsize=(16, 12))
        top_features = feature_importance['Variable'].head(10).tolist()
        sns.heatmap(self.train_data[top_features].corr(), 
                   annot=True, cmap='coolwarm', center=0)
        plt.title('Correlaciones entre Variables Principales')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('../visualizaciones/16_correlacion_top_variables.png')
        plt.close()
        
        # 17. Scatter plot PCA
        plt.figure(figsize=(12, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
        plt.xlabel('Primer Componente Principal')
        plt.ylabel('Segundo Componente Principal')
        plt.title('Visualización PCA - Primeros 2 Componentes')
        plt.colorbar(label='Bancarrota')
        plt.savefig('../visualizaciones/17_pca_scatter.png')
        plt.close()
    
    def ejecutar_analisis_completo(self):
        """Executa todas as análises em sequência"""
        self.analisis_exploratorio()
        self.analisis_avanzado()
        self.analisis_completo()
        
        print("\nANÁLISIS COMPLETO FINALIZADO")
        print("-" * 50)
        print("\nVisualizaciones guardadas en el directorio 'visualizaciones':")
        print("- 01_distribucion_bancarrota.png")
        print("- 02_beneficio_operativo.png")
        print("- 03_pasivo_corriente.png")
        print("- 04_correlacion_inicial.png")
        print("- 05_ratio_deuda.png")
        print("- 06_distribucion_roa.png")
        print("- 07_top_correlaciones.png")
        print("- 08_distribucion_top_variable.png")
        print("- 09_scatter_correlaciones.png")
        print("- 10_densidad_beneficio.png")
        print("- 11_violin_var1.png")
        print("- 12_violin_var2.png")
        print("- 13_pca_varianza.png")
        print("- 14_importancia_variables.png")
        print("- 15_distribucion_top_mi.png")
        print("- 16_correlacion_top_variables.png")
        print("- 17_pca_scatter.png")
        
        print("\nConclusiones Principales:")
        print("1. Conjunto de datos altamente desbalanceado")
        print("2. Variables más importantes identificadas")
        print("3. Diferencias significativas entre grupos")
        print("4. Alta dimensionalidad con potencial de reducción")
        
        print("\nRecomendaciones para Modelado:")
        print("1. Usar técnicas de balance de clases")
        print("2. Considerar transformación de variables")
        print("3. Evaluar uso de PCA")
        print("4. Enfocarse en variables más importantes")

if __name__ == "__main__":
    analisis = AnalisisBancarrota()
    analisis.ejecutar_analisis_completo() 