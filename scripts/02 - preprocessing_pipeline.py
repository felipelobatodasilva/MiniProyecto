import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

class PreprocessingPipeline:
    def __init__(self):
        self.numeric_features = None
        self.target = 'Bankruptcy'
        self.id_column = 'ID'
        
    def fit_transform(self, train_data, test_data=None):
        """
        Aplica o pipeline de pré-processamento aos dados de treino e teste.
        
        Parameters:
        -----------
        train_data : pandas.DataFrame
            Dados de treinamento
        test_data : pandas.DataFrame, optional
            Dados de teste
            
        Returns:
        --------
        X_train_transformed : numpy.ndarray
            Features de treino transformadas
        y_train : numpy.ndarray
            Target de treino
        X_test_transformed : numpy.ndarray, optional
            Features de teste transformadas
        """
        print("Iniciando pipeline de pré-processamento...")
        
        # 1. Separar features e target
        print("\n1. Separando features e target...")
        X_train, y_train = self._separate_features_target(train_data, is_train=True)
        if test_data is not None:
            X_test, _ = self._separate_features_target(test_data, is_train=False)
        
        # 2. Identificar tipos de features
        print("\n2. Identificando tipos de features...")
        self._identify_feature_types(X_train)
        
        # 3. Criar pipeline de transformação
        print("\n3. Criando pipeline de transformação...")
        preprocessing_pipeline = self._create_preprocessing_pipeline()
        
        # 4. Aplicar transformações
        print("\n4. Aplicando transformações...")
        X_train_transformed = preprocessing_pipeline.fit_transform(X_train)
        
        # 5. Aplicar SMOTE para balanceamento
        print("\n5. Aplicando SMOTE para balanceamento...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_transformed, y_train)
        
        print("\nDimensões após transformação:")
        print(f"X_train: {X_train_balanced.shape}")
        print(f"y_train: {y_train_balanced.shape}")
        
        if test_data is not None:
            X_test_transformed = preprocessing_pipeline.transform(X_test)
            print(f"X_test: {X_test_transformed.shape}")
            return X_train_balanced, y_train_balanced, X_test_transformed
        
        return X_train_balanced, y_train_balanced
    
    def _separate_features_target(self, data, is_train=True):
        """Separa features e target do DataFrame."""
        if is_train:
            X = data.drop([self.target, self.id_column], axis=1)
            y = data[self.target]
        else:
            X = data.drop([self.id_column], axis=1)
            y = None
        return X, y
    
    def _identify_feature_types(self, X):
        """Identifica os tipos de features no dataset."""
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        print("Features numéricas identificadas:")
        print(f"Total: {len(self.numeric_features)}")
    
    def _create_preprocessing_pipeline(self):
        """Cria o pipeline de pré-processamento."""
        # Pipeline para features com muitos outliers
        high_outlier_features = [
            'Operating.Expense.Rate',
            'Quick.Asset.Turnover.Rate',
            'Fixed.Assets.Turnover.Frequency',
            'Current.Asset.Turnover.Rate',
            'Total.Asset.Growth.Rate'
        ]
        
        # Pipeline para features importantes
        important_features = [
            'Tax.rate..A.',
            'Working.Capital.to.Total.Assets',
            'Total.debt.Total.net.worth',
            'Retained.Earnings.to.Total.Assets',
            'Pre.tax.net.Interest.Rate'
        ]
        
        # Outras features numéricas
        other_numeric = [f for f in self.numeric_features 
                        if f not in high_outlier_features + important_features]
        
        # Criar transformadores
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        robust_transformer = Pipeline(steps=[
            ('scaler', RobustScaler())
        ])
        
        # Combinar transformadores
        preprocessor = ColumnTransformer(
            transformers=[
                ('robust', robust_transformer, high_outlier_features),
                ('standard', numeric_transformer, important_features + other_numeric)
            ])
        
        return preprocessor

if __name__ == "__main__":
    # Carregar dados
    print("Cargando datos...")
    train_data = pd.read_csv('../archivos/train_data.csv')
    test_data = pd.read_csv('../archivos/test_data.csv')
    
    # Criar e aplicar pipeline
    print("Iniciando pipeline de preprocesamiento...")
    pipeline = PreprocessingPipeline()
    X_train, y_train, X_test = pipeline.fit_transform(train_data, test_data)
    
    # Salvar dados processados
    print("Guardando datos procesados...")
    np.save('X_train_processed.npy', X_train)
    np.save('y_train_processed.npy', y_train)
    np.save('X_test_processed.npy', X_test)
    
    print("Preprocesamiento completado exitosamente!")
    print("\nDimensões finais:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}") 