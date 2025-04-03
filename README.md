# Proyecto de Predicción de Bancarrota Empresarial

## Índice
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Preparación del Ambiente](#preparación-del-ambiente)
- [Resumen](#resumen)
- [Contexto](#contexto)
- [Exploración de Datos](#1-exploración-de-datos)
- [Preparación de Datos](#2-preparación-de-datos)
- [Análisis Preliminar y Selección de Modelos](#3-análisis-preliminar-y-selección-de-modelos)
- [Desarrollo y Calibración de Modelos](#4-desarrollo-y-calibración-de-modelos)
- [Visualización de Resultados](#5-visualización-de-resultados)
- [Participación en Kaggle](#6-participación-en-kaggle)
- [Métrica de Evaluación](#métrica-de-evaluación)
- [Formato de Submisión](#formato-de-submisión)
- [Criterios de evaluación](#criterios-de-evaluación)

## Estructura del Proyecto

```
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── 01_analisis_completo.py
│   ├── 02_preprocessing_pipeline.py
│   ├── 03_model_training.py
│   ├── 01_analisis_completo_notebook.py
│   ├── 02_preprocessing_pipeline_notebook.py
│   └── 03_model_training_notebook.py
├── archivos/
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── sampleSubmission.csv
│   ├── submission.csv
│   ├── X_train_processed.npy
│   ├── y_train_processed.npy
│   └── X_test_processed.npy
├── visualizaciones/
│   ├── 01_distribucion_bancarrota.png
│   ├── 02_beneficio_operativo.png
│   ├── 03_pasivo_corriente.png
│   ├── 04_correlacion_inicial.png
│   ├── 05_ratio_deuda.png
│   ├── 06_distribucion_roa.png
│   ├── 07_top_correlaciones.png
│   ├── 08_distribucion_top_variable.png
│   ├── 09_scatter_correlaciones.png
│   ├── 10_densidad_beneficio.png
│   ├── 11_violin_var1.png
│   ├── 12_violin_var2.png
│   ├── 13_pca_varianza.png
│   ├── 14_importancia_variables.png
│   ├── 15_distribucion_top_mi.png
│   ├── 16_correlacion_top_variables.png
│   └── 17_pca_scatter.png
├── visualizaciones1/
│   └── (mesmas visualizaciones que visualizaciones/)
├── visualizaciones_modelos/
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── training_history.png
│   └── probability_distribution.png
└── venv/
```

## Preparación del Ambiente

Para ejecutar los códigos de este proyecto, siga los siguientes pasos:

1. **Crear y activar entorno virtual**:
   ```bash
   # Crear entorno virtual
   python -m venv venv

   # Activar entorno virtual
   # En Linux/Mac:
   source venv/bin/activate
   # En Windows:
   # venv\Scripts\activate
   ```

2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verificar instalación**:
   ```bash
   pip list
   ```
   Debe ver las siguientes bibliotecas con sus versiones exactas:
   - pandas==2.2.3
   - numpy==2.2.4
   - matplotlib==3.10.1
   - seaborn==0.13.2
   - scipy==1.15.2
   - scikit-learn==1.6.1
   - imbalanced-learn==0.13.0
   - xgboost==3.0.0

4. **Ejecutar los scripts**:
   Ejecute los scripts en el siguiente orden:
   ```bash
   # 1. Análisis exploratorio
   python scripts/01_analisis_completo.py

   # 2. Preprocesamiento
   python scripts/02_preprocessing_pipeline.py

   # 3. Entrenamiento y evaluación
   python scripts/03_model_training.py
   ```

## Resumen

En este proyecto pondrán en práctica la identificación e implementación de redes neuronales en modelos predictivos que sean pertinentes para resolver una situación problemática asociada a un conjunto de datos. Se busca que puedan reconocer las ventajas y limitaciones de dichos modelos y comunicar efectivamente los resultados de su implementación.

## Contexto

¿Es posible predecir con anterioridad cuándo una empresa comercial se declarará en bancarrota? Una respuesta positiva a esta pregunta permitiría a bancos y acreedores tomar acciones para proteger sus intereses, y para los demás participantes del mercado obtener primas por riesgo de crédito más justas.

Si bien existen muchas señales que pueden indicar que una empresa irá a bancarrota, mucha de la información disponible proviene de la misma organización, con lo cual se crean asimetrías de información e incentivos perversos. Por esto, es importante poder predecir este fenómeno a partir de información pública de cada empresa que esté obligada a reportar, como los estados financieros, a partir de los cuales, es posible calcular muchas razones financieras que explican el comportamiento de la empresa.

En este proyecto, se utilizan datos reales para predecir el si una empresa se declarará en bancarrota en el año siguiente a la observación de su información financiera. Para cada empresa, se tienen en total 63 razones financieras calculadas y la muestra corresponde a compañías en China durante los años de 1999 a 2009 que aparecen en el Shanghai Stock Exchange.

## 1. Exploración de Datos
**Objetivo**: Comprender el conjunto de datos dentro del contexto organizacional (empresas que pueden declararse en bancarrota).

**Qué hemos hecho**:
- Cargado e inspeccionado los conjuntos de datos de entrenamiento y prueba
- Generado estadísticas descriptivas para cada variable
- Elaborado histogramas, boxplots y correlogramas
- Identificado comportamientos, valores atípicos y relaciones entre variables
- Justificado la importancia de cada variable para la predicción de bancarrota

## 2. Preparación de Datos
**Objetivo**: Transformar los datos en bruto en un formato adecuado para los modelos de redes neuronales.

**Qué hemos hecho**:
- Implementado pipeline de preprocesamiento
- Normalizado datos usando StandardScaler y RobustScaler
- Tratado outliers de manera apropiada
- Aplicado SMOTE para balancear las clases
- Mantenido consistencia en el preprocesamiento entre conjuntos de entrenamiento y prueba

## 3. Análisis Preliminar y Selección de Modelos
**Objetivo**: Justificar la elección de modelos de redes neuronales para la predicción de bancarrota.

**Qué hemos hecho**:
- Seleccionado Perceptrón Multicapa (MLP) como modelo principal
- Justificado la elección basada en las características del problema
- Implementado validación cruzada para evaluación

## 4. Desarrollo y Calibración de Modelos
**Objetivo**: Construir, entrenar y ajustar los modelos elegidos.

**Qué hemos hecho**:
- Implementado red neuronal MLP con arquitectura (100, 50)
- Utilizado función de activación ReLU
- Implementado early stopping para evitar sobreajuste
- Ajustado hiperparámetros mediante validación cruzada
- Calibrado el modelo para mejorar su desempeño

## 5. Visualización de Resultados
**Objetivo**: Comunicar de forma clara los hallazgos y conclusiones.

**Qué hemos hecho**:
- Generado curvas ROC y Precision-Recall
- Creado matriz de confusión
- Visualizado histórico de entrenamiento
- Analizado distribución de probabilidades
- Interpretado resultados en el contexto del problema

## 6. Participación en Kaggle
**Objetivo**: Subir las predicciones y participar en la competencia.

**Qué hemos hecho**:
- Generado predicciones en el formato requerido
- Creado archivo de submissión con columnas ID y Bankruptcy
- Preparado para subir resultados a la competencia

## Métrica de Evaluación

La métrica de evaluación para esta competencia es AUC (Area Under the Curve). El AUC es una métrica de evaluación comúnmente utilizada para problemas binarios como este. La interpretación es que, dado un caso positivo y uno negativo aleatorios, el AUC da la proporción de veces que se adivina correctamente cuál es cuál. Es menos afectado por el balance de muestras que la precisión. Un modelo perfecto obtendrá un AUC de 1, mientras que una predicción aleatoria obtendrá un AUC de alrededor de 0.5.

## Formato de Submisión

Para cada empresa en el conjunto de datos, los archivos de envío deben contener dos columnas: ID y Bankruptcy. El archivo de envío debe ser un archivo CSV. El ID debe ser simplemente la columna de ID de observación correspondiente del conjunto de datos. La columna Bankruptcy debe ser la probabilidad predicha del resultado 1 para ese ID de observación.

El archivo debe contener un encabezado y tener el siguiente formato:

```
ID,Bankruptcy
1,0.8
2,0.3
3,0.6
```

## Criterios de evaluación

Para este miniproyecto, deben entregar un informe en formato PDF que incluya los códigos y análisis realizados. Para el desarrollo de cada uno de los apartados del proyecto, deberán tener en cuenta los siguientes criterios de evaluación:

1. **Exploración de los datos para su entendimiento dentro del contexto organizacional** [10 puntos]
   - Se utilizan histogramas, correlogramas y estadísticas descriptivas para la exploración preliminar de los datos del problema
   - Se deja claro el rol que cumple cada una de las variables a utilizar
   - Se argumenta la razón por la que el uso de estas variables puede contribuir a la solución del problema por medio de modelos predictivos

2. **Preparación de los datos para poder utilizarlos como entrada para modelos predictivos** [10 puntos]
   - Se utilizan correctamente los procedimientos de preprocesamiento de datos vistos en el curso para la preparación de los datos antes de la implementación de los modelos predictivos

3. **Análisis preliminar de selección de modelos relevantes para responder a la pregunta** [20 puntos]
   - De acuerdo a la definición del problema predictivo planteado, se argumenta qué modelos predictivos son candidatos a utilizarse para solucionar la pregunta de interés en el respectivo contexto organizacional

4. **Desarrollo y calibración de modelos** [40 puntos]
   - Los modelos escogidos como candidatos son calibrados y utilizados de manera correcta

5. **Visualización de resultados** [20 puntos]
   - Se presenta un análisis gráfico completo de los resultados que permite entender la manera en la que los modelos implementados responden a la pregunta de interés

6. **Puntos adicionales**
   - El mejor modelo predictivo recibirá 10 puntos adicionales sobre la nota final y los siguientes puestos recibirán puntos parciales
   - Las posiciones de los equipos se definirán en la competencia disponible en Kaggle
   - Nota: Kaggle genera 2 listas de posiciones:
     - Una pública disponible durante la competencia (70% de los resultados)
     - Una privada disponible una vez finalice la competencia (30% de los resultados)
   - La posición final de cada grupo será la posición ponderada entre ambas listas

## Enlace a la competencia
https://www.kaggle.com/competitions/prediccion-de-bancarrota-empresaria
