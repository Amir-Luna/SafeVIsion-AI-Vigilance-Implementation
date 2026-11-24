# SafeVision AI — Verificación Inteligente de Uso de EPP

Sistema modular de Visión Computacional e Inteligencia Artificial desarrollado para el **Laboratorio de Mecanizado de Mecatrónica**, orientado a verificar el uso **correcto** del Equipamiento de Protección Personal (EPP) en dos actividades:

- **Corte**: gafas de seguridad + guantes de corte en ambas manos.
- **Soldadura**: máscara de soldadura bajada + guantes de soldadura en ambas manos.

*(La evaluación de vestimenta —camisa y pantalón jean— queda preparada como ampliación futura.)*

---

## 1. Arquitectura general de SafeVision AI

SafeVision AI opera en dos etapas coordinadas por un motor de decisión desacoplado (preparado para integrarse posteriormente en una interfaz gráfica):

1. **Modelo 1 — Detector YOLO (Ultralytics con Transfer Learning)**: localiza personas, regiones corporales de referencia (`CABEZA`, `MANO`) y objetos EPP independientemente de cómo estén colocados.
2. **Control de calidad visual + Recorte contextual**: cada región `CABEZA` y `MANO` se recorta con margen configurable y se evalúa su resolución, nitidez (varianza del Laplaciano) e iluminación.
3. **Modelo 2 — CNN propia en PyTorch (`SafeVisionCNN`, entrenada desde cero)**: analiza cada recorte corporal (224 × 224 RGB con *letterbox* sin deformación) para clasificar si el EPP obligatorio está **correctamente colocado** o **ausente / mal colocado / sostenido**.
4. **Motor de decisión (`src/decision_engine.py`)**: emite `CUMPLE`, `NO_CUMPLE` o `NO_VERIFICABLE` aplicando las reglas de cada actividad.

## 2. Función de YOLO

YOLO utiliza pesos preentrenados oficiales (por defecto `yolo11n.pt`, configurable en `configs/config.yaml`) únicamente como inicialización y se ajusta mediante **Transfer Learning** exclusivamente con su dataset propio.

| ID | Clase | Rol en el sistema |
|---|---|---|
| 0 | `PERSONA` | Contexto general de la persona visible |
| 1 | `CABEZA` | Región de referencia para gafas o máscara |
| 2 | `MANO` | Región de referencia para guantes (cada mano individual) |
| 3 | `GAFAS_SEGURIDAD` | Detección auxiliar de gafas (puestas, en frente o sostenidas) |
| 4 | `GUANTES_CORTE` | Detección auxiliar de guantes de corte |
| 5 | `MASCARA_SOLDADURA` | Detección auxiliar de máscara (bajada o levantada) |
| 6 | `GUANTES_SOLDADURA` | Detección auxiliar de guantes de soldadura |

> **Regla clave de anotación**: si una mano lleva guante, se anotan **ambas cajas** (`MANO` y el guante correspondiente, aunque se superpongan). Si la mano está descubierta, se anota únicamente `MANO`. Lo mismo aplica a `CABEZA`.

## 3. Función de la CNN propia

`SafeVisionCNN` (`src/classifier.py`) es una red convolucional implementada desde cero en PyTorch (sin pesos preentrenados ni transfer learning), con inicialización Kaiming/Xavier, 5 bloques `Conv2d + BatchNorm2d + ReLU + MaxPool2d`, `AdaptiveAvgPool2d`, `Dropout` y capas totalmente conectadas hacia **8 clases**:

| ID | Clase CNN | Región | Actividad |
|---|---|---|---|
| 0 | `GAFAS_COLOCADAS` | `CABEZA` | `corte` |
| 1 | `GAFAS_NO_COLOCADAS` | `CABEZA` | `corte` |
| 2 | `GUANTES_CORTE_COLOCADOS` | `MANO` | `corte` |
| 3 | `GUANTES_CORTE_NO_COLOCADOS` | `MANO` | `corte` |
| 4 | `MASCARA_COLOCADA` | `CABEZA` | `soldadura` |
| 5 | `MASCARA_NO_COLOCADA` | `CABEZA` | `soldadura` |
| 6 | `GUANTES_SOLDADURA_COLOCADOS` | `MANO` | `soldadura` |
| 7 | `GUANTES_SOLDADURA_NO_COLOCADOS` | `MANO` | `soldadura` |

## 4. Estructura de carpetas y datasets

```text
SafeVision_AI/
├── configs/
│   ├── config.yaml              # Parámetros globales (umbrales, rutas, hiperparámetros)
│   └── data.yaml                # Definición del dataset y 7 clases YOLO
├── datasets/
│   ├── original/
│   │   ├── images/              # Fotografías originales completas
│   │   ├── labels/              # Anotaciones YOLO (.txt) de cada foto original
│   │   └── metadata.csv         # Relación foto -> actividad, grupo_sesion, fuente, division
│   ├── yolo/
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/
│   └── cnn/
│       ├── revision_pendiente/  # Recortes pendientes y etiquetas_revision.csv
│       ├── train/<8_clases>/
│       ├── val/<8_clases>/
│       └── test/<8_clases>/
├── models/
│   ├── yolo/                    # Aquí se guardará best.pt tras entrenar YOLO
│   └── cnn/                     # Aquí se guardará best_cnn.pth tras entrenar la CNN
├── outputs/
│   └── evaluation/              # Reportes, matrices de confusión y métricas
├── src/
│   ├── detector.py
│   ├── classifier.py
│   ├── decision_engine.py
│   ├── camera.py
│   └── utils.py
├── scripts/
│   ├── train_yolo.py
│   ├── generar_dataset_cnn.py
│   ├── train_cnn.py
│   └── evaluate_models.py
├── tests/
│   └── test_safevision.py
├── main.py
├── requirements.txt
└── README.md
```

## 5. Formato de anotaciones YOLO

Cada fotografía `nombre.jpg` debe tener su archivo `nombre.txt` con una línea por objeto:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Todos los valores de coordenadas están normalizados en `[0.0, 1.0]`. Ejemplo:

```text
0 0.5020 0.5410 0.4200 0.8900
1 0.4950 0.2100 0.1600 0.1900
3 0.4960 0.1950 0.1200 0.0600
2 0.3400 0.6100 0.1100 0.1300
4 0.3420 0.6080 0.1050 0.1250
```

## 6. Cómo organizar las fotografías originales

1. Coloque todas sus fotografías originales en `datasets/original/images/`.
2. Coloque los archivos `.txt` anotados (con exactamente el mismo nombre base) en `datasets/original/labels/`.
3. Para entrenar YOLO, divida sus fotografías y etiquetas en `datasets/yolo/images/{train,val,test}` y `datasets/yolo/labels/{train,val,test}` manteniendo las fotos de una misma persona/sesión en el mismo conjunto.

## 7. Cómo crear `metadata.csv`

Edite `datasets/original/metadata.csv` (sin líneas comentadas) respetando las columnas:

```csv
archivo,actividad,grupo_sesion,fuente,division
img_001.jpg,corte,sesion_01,webcam,
img_002.jpg,corte,sesion_01,webcam,
img_003.jpg,soldadura,sesion_02,celular,
img_004.jpg,soldadura,sesion_03,webcam,test
```

- `actividad`: `corte` o `soldadura`.
- `grupo_sesion`: identificador de la persona o sesión de captura (garantiza que los recortes de una misma sesión nunca se mezclen entre `train`, `val` y `test`).
- `fuente`: `webcam`, `celular` o `internet`.
- `division`: opcional (`train`, `val`, `test`); si se deja vacía, el script asigna automáticamente 70 % / 15 % / 15 % por grupo.

---

## Instalación en Windows (CMD)

Asegúrese de tener **Python 3.11** instalado y agregado al `PATH`. Abra el **Símbolo del sistema (CMD)** en la carpeta `SafeVision_AI`:

```bat
cd SafeVision_AI
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

### Instalación de PyTorch y TorchVision

- **Con GPU NVIDIA (CUDA 12.1 — recomendado si cuenta con GPU compatible):**

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

- **Solo CPU (sin GPU NVIDIA):**

```bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Instalación de Ultralytics y demás librerías

```bat
pip install ultralytics
pip install -r requirements.txt
```

Verifique la detección de GPU:

```bat
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA disponible:', torch.cuda.is_available())"
```

---

## 8. Cómo entrenar YOLO

Con el entorno virtual activado y su dataset colocado en `datasets/yolo/`:

```bat
python scripts\train_yolo.py --epochs 100 --imgsz 640 --batch 16
```

Al finalizar, el mejor modelo se guardará automáticamente en `models\yolo\best.pt` y las métricas/gráficas en `outputs\runs\yolo_safevision\`.

## 9. Cómo generar los recortes de la CNN

```bat
python scripts\generar_dataset_cnn.py --mode extract
```

Esto genera los recortes de `CABEZA` y `MANO` en `datasets\cnn\revision_pendiente\` y crea `etiquetas_revision.csv`.

## 10. Cómo revisar y clasificar manualmente los recortes

```bat
python scripts\generar_dataset_cnn.py --mode review
```

Controles en la ventana OpenCV:
- **`1`**: uso correcto (asigna automáticamente la clase correspondiente según actividad y región).
- **`2`**: uso incorrecto / ausente / sostenido.
- **`3`**: no verificable / descartar.
- **`N`**: saltar a la siguiente imagen sin clasificar.
- **`Q`**: guardar progreso y salir (puede reanudar después).

Una vez revisados los recortes, construya el dataset definitivo 70/15/15 por grupo:

```bat
python scripts\generar_dataset_cnn.py --mode build
```

## 11. Cómo entrenar la CNN propia (desde cero)

```bat
python scripts\train_cnn.py --epochs 60 --batch-size 32 --lr 0.001
```

El mejor modelo se guardará en `models\cnn\best_cnn.pth` (incluyendo arquitectura, mapa de clases y parámetros de normalización). Las curvas de entrenamiento, el reporte de clasificación y la matriz de confusión de validación se guardan en `outputs\runs\cnn_training\`.

## 12. Cómo evaluar ambos modelos

Evaluación completa (YOLO en test, CNN en test y pipeline conjunto):

```bat
python scripts\evaluate_models.py --mode all
```

También puede evaluar cada componente por separado (`--mode yolo`, `--mode cnn`, `--mode joint`). Los resultados se guardan en `outputs\evaluation\`.

## 13. Cómo ejecutar SafeVision AI en tiempo real

- **Webcam (actividad corte):**
```bat
python main.py --activity corte --source 0
```

- **Webcam (actividad soldadura):**
```bat
python main.py --activity soldadura --source 0
```

- **Archivo de video o fotografía:**
```bat
python main.py --activity corte --source ruta\al\video.mp4
python main.py --activity soldadura --source ruta\a\foto.jpg
```

Presione **`Q`** en la ventana de visualización para cerrar correctamente.

## Ejecutar pruebas unitarias

```bat
python -m unittest discover -s tests -v
```
