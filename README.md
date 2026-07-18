# PSIB 2026 Q1 ITBA — Trabajo Final — Grupo 11

## Clasificación de lesiones mamarias ecográficas utilizando máscaras manuales y automáticas

Integrantes: Cobián, Perelstein y Obeid.

## Objetivo

El proyecto estudia la clasificación binaria de lesiones mamarias ecográficas
en las clases `benign` y `malignant` utilizando el dataset BUSI.

Se comparan dos fuentes de segmentación:

1. Máscaras manuales de referencia.
2. Máscaras automáticas producidas por una U-Net.

Sobre cada fuente se evaluaron dos estrategias de clasificación:

1. Características diseñadas manualmente, PCA y SVM-RBF.
2. Transfer learning con EfficientNet-B0 sobre la región segmentada.

El objetivo central es analizar cuánto se modifica el desempeño de los
clasificadores cuando las máscaras manuales son reemplazadas por máscaras
automáticas.

---

## Metodología general

El flujo del proyecto es:

```text
Dataset BUSI
    ↓
Exploración y preprocesamiento
    ↓
Split reproducible mediante manifest.csv
    ↓
Segmentación U-Net
    ↓
Máscaras manuales y automáticas
    ↓
Extracción de 53 características
    ↓
Alineación de cohortes
    ↓
PCA ajustado exclusivamente con train
    ↓
SVM-RBF
    ↓
EfficientNet-B0
    ↓
Comparación final sobre el mismo conjunto de test
```

Las 53 características pertenecen a cuatro familias:

- Intensidad.
- Morfología.
- Textura GLCM/Haralick.
- Wavelets 2D.

---

## Separación de los datos

Se utilizaron únicamente las clases benignas y malignas.

El conjunto disponible se dividió inicialmente en dos grupos estratificados:

### Grupo U-Net

- Train: 267 imágenes.
- Validation: 48 imágenes.
- Total: 315 imágenes.

Estas imágenes se utilizaron exclusivamente para entrenar y seleccionar la
segmentación U-Net.

### Grupo de clasificadores

- Train: 219 imágenes.
- Validation: 48 imágenes.
- Test: 48 imágenes.
- Total: 315 imágenes.

La U-Net nunca utilizó imágenes del grupo de clasificadores durante su
entrenamiento.

El split está fijado en:

```text
data/splits/manifest.csv
```

Este archivo no debe regenerarse con otra semilla.

---

## Cohorte común manual–automática

Algunas máscaras automáticas vacías o degeneradas produjeron características
no finitas.

Para realizar una comparación justa, se conservaron solamente las imágenes
con características válidas tanto para la máscara manual como para la
automática.

La cohorte alineada final contiene:

- Train: 216 imágenes.
- Validation: 48 imágenes.
- Test: 47 imágenes.
- Total: 311 imágenes.

Los cuatro casos excluidos y sus motivos quedan documentados en:

```text
outputs/tables_aligned/excluded_cases.csv
```

Las tablas alineadas utilizadas por PCA y los clasificadores son:

```text
outputs/tables_aligned/features_manual.csv
outputs/tables_aligned/features_auto.csv
```

---

## Prevención de fuga de datos

Se aplicaron las siguientes reglas metodológicas:

- El `StandardScaler` y el PCA se ajustaron solamente con `train`.
- Validation y test fueron transformados con los objetos ajustados en train.
- Los hiperparámetros de la SVM se seleccionaron mediante validación cruzada
  dentro de train.
- Validation se utilizó para comparar configuraciones de EfficientNet.
- Test no se utilizó para seleccionar hiperparámetros.
- Test se evaluó únicamente en las corridas finales.
- Los cuatro modelos se compararon sobre las mismas 47 imágenes de test.

El PCA retuvo aproximadamente el 95 % de la varianza:

- Máscaras manuales: 16 componentes principales.
- Máscaras automáticas: 15 componentes principales.

---

## Estructura relevante del repositorio

```text
data/
├── raw/                         # Dataset original; no versionado
├── processed/
│   ├── preprocessed/            # Imágenes preprocesadas; no versionadas
│   └── auto_masks/              # Máscaras automáticas versionadas
└── splits/
    └── manifest.csv             # Split oficial del proyecto

scripts/
├── 00_split_dataset.py
├── 01_data_loading_exploration.py
├── 02_select_and_characterize_subset.py
├── 03_preprocessing_comparison.py
├── 04_segmentation_GUI_*.py
├── 05_texture_glcm_features.py
├── 06_lesion_ecographic_characterization.py
├── 07_single_case_feature_viewer_GUI.py
├── 09_unet_inference.py
├── 09b_summarize_unet_metrics.py
├── 10_extract_features.py
├── 10b_align_feature_tables.py
├── 11_pca_analysis.py
├── 12_train_evaluate_svm.py
├── 13_train_efficientnet.py
└── 14_compare_models.py

outputs/
├── efficientnet/
├── figures/
├── models/
├── tables/
└── tables_aligned/
```

Los módulos 01 a 07 corresponden a la exploración, preprocesamiento,
segmentación clásica y caracterización desarrollados durante las etapas
anteriores del proyecto.

La clasificación final reproducible comienza con las máscaras manuales y
automáticas ya disponibles y continúa con los módulos 10 a 14.

---

## Instalación

### 1. Crear el entorno virtual

En PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy `
  -Scope Process `
  -ExecutionPolicy Bypass
```

### 2. Instalar dependencias

```powershell
python -m pip install -r requirements.txt
```

Para utilizar la configuración CUDA empleada durante el proyecto puede
instalarse PyTorch con CUDA 11.8 antes del resto de las dependencias:

```powershell
python -m pip install `
  torch torchvision `
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install -r requirements.txt
```

### 3. Verificar el entorno

```powershell
python -m pip check
```

```powershell
python -c "import numpy, pandas, matplotlib, cv2, skimage, scipy, PySide6, pywt, sklearn, joblib, PIL, torch, torchvision, albumentations, segmentation_models_pytorch; print('Dependencias OK')"
```

Comprobar PyTorch y CUDA:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available()); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Datos necesarios

El dataset BUSI original debe ubicarse bajo:

```text
data/raw/
```

Las carpetas pesadas de datos crudos y preprocesados no se incluyen en Git.

Las máscaras automáticas utilizadas en los resultados finales sí se
encuentran versionadas en:

```text
data/processed/auto_masks/
```

El checkpoint de U-Net no se incluye porque es un archivo binario
regenerable. Por ese motivo, el módulo 09 solamente debe ejecutarse cuando se
disponga localmente de un checkpoint compatible.

---

## Orden de ejecución de la etapa final

Los resultados finales ya están versionados. Los siguientes comandos se
incluyen para reproducirlos desde las tablas y máscaras disponibles.

### 1. Resumir la segmentación U-Net

```powershell
python scripts\09b_summarize_unet_metrics.py
```

Salidas:

```text
outputs/tables/09_unet_metrics_summary.csv
outputs/figures/unet/09_unet_metrics_distribution.png
```

### 2. Extraer características

```powershell
python scripts\10_extract_features.py
```

Salidas:

```text
outputs/tables/features_manual.csv
outputs/tables/features_auto.csv
```

### 3. Construir la cohorte alineada

```powershell
python scripts\10b_align_feature_tables.py
```

Salidas:

```text
outputs/tables_aligned/features_manual.csv
outputs/tables_aligned/features_auto.csv
outputs/tables_aligned/excluded_cases.csv
```

### 4. Ejecutar PCA sobre las tablas alineadas

```powershell
python scripts\11_pca_analysis.py `
  --tables-dir outputs\tables_aligned `
  --output-tables-dir outputs\tables_aligned `
  --figures-dir outputs\figures\pca_aligned `
  --models-dir outputs\models\pca_aligned
```

El scaler y el PCA se ajustan exclusivamente con las filas de train.

### 5. Entrenar y evaluar las SVM-RBF

```powershell
python scripts\12_train_evaluate_svm.py
```

Este módulo:

- selecciona `C` y `gamma` mediante validación cruzada en train;
- informa resultados de validation;
- reajusta el modelo final con train + validation;
- evalúa test una sola vez.

### 6. Seleccionar EfficientNet mediante validation

Configuraciones manuales:

```powershell
python scripts\13_train_efficientnet.py `
  --mask-source manual `
  --run-name manual_d03_p015 `
  --dropout 0.3 `
  --padding 0.15
```

```powershell
python scripts\13_train_efficientnet.py `
  --mask-source manual `
  --run-name manual_d04_p020 `
  --dropout 0.4 `
  --padding 0.20
```

Configuraciones automáticas:

```powershell
python scripts\13_train_efficientnet.py `
  --mask-source auto `
  --run-name auto_d03_p015 `
  --dropout 0.3 `
  --padding 0.15
```

```powershell
python scripts\13_train_efficientnet.py `
  --mask-source auto `
  --run-name auto_d04_p020 `
  --dropout 0.4 `
  --padding 0.20
```

Estas corridas no evalúan test.

La comparación de validation se guarda en:

```text
outputs/tables/13_efficientnet_validation_comparison.csv
```

La configuración elegida para ambas fuentes fue:

```text
dropout = 0.3
padding = 0.15
```

### 7. Ejecutar las corridas finales de EfficientNet

Máscara manual:

```powershell
python scripts\13_train_efficientnet.py `
  --mask-source manual `
  --run-name final_manual `
  --dropout 0.3 `
  --padding 0.15 `
  --evaluate-test
```

Máscara automática:

```powershell
python scripts\13_train_efficientnet.py `
  --mask-source auto `
  --run-name final_auto `
  --dropout 0.3 `
  --padding 0.15 `
  --evaluate-test
```

### 8. Comparar los cuatro modelos

```powershell
python scripts\14_compare_models.py
```

El script verifica que las cuatro evaluaciones contengan exactamente los
mismos filenames de test.

Salidas:

```text
outputs/tables/14_model_comparison.csv
outputs/figures/comparison/14_model_comparison.png
```

---

## Resultados finales

Resultados sobre la cohorte común de 47 imágenes de test:

| Modelo | Máscara | Accuracy | Balanced accuracy | Sensibilidad | Especificidad | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| SVM-RBF | Manual | 0.9574 | 0.9677 | 1.0000 | 0.9355 | 0.9412 | 1.0000 |
| EfficientNet-B0 | Manual | 0.9574 | 0.9526 | 0.9375 | 0.9677 | 0.9375 | 0.9940 |
| SVM-RBF | Automática | 0.8511 | 0.8871 | 1.0000 | 0.7742 | 0.8205 | 0.9214 |
| EfficientNet-B0 | Automática | 0.8085 | 0.7944 | 0.7500 | 0.8387 | 0.7273 | 0.8629 |

La tabla fuente se encuentra en:

```text
outputs/tables/14_model_comparison.csv
```

---

## Interpretación principal

Con máscaras manuales, SVM-RBF y EfficientNet-B0 alcanzaron desempeños
elevados y similares.

Las máscaras automáticas redujeron el desempeño de ambos métodos. La
degradación fue menor para la SVM-RBF, que mantuvo una balanced accuracy de
0.8871, mientras que EfficientNet-B0 obtuvo 0.7944.

Esto indica que, dentro de esta cohorte, las características diseñadas
manualmente y la SVM fueron más robustas frente a errores de segmentación que
el clasificador basado directamente en la región recortada.

---

## Limitaciones

- El conjunto de test contiene solamente 47 imágenes.
- La clase maligna es minoritaria.
- Las métricas pueden presentar alta variabilidad debido al tamaño de la
  muestra.
- Un ROC-AUC de 1.0 en una cohorte pequeña no implica desempeño perfecto en
  población externa.
- No se realizó validación externa con otro centro o equipo ecográfico.
- La calidad irregular de algunas máscaras automáticas afecta la extracción
  de características y el recorte utilizado por EfficientNet.
- Los resultados no deben interpretarse como un sistema de diagnóstico
  clínico validado.

---

## Archivos binarios

Los siguientes archivos no se versionan porque son regenerables:

```text
*.pt
*.joblib
```

Esto incluye checkpoints de U-Net, EfficientNet, PCA y SVM.

Los parámetros, configuraciones, predicciones, métricas, tablas y figuras sí
se conservan en el repositorio.