# TextLab — Cuaderno de experimentación NLP

TextLab es una aplicación web para realizar experimentos de **clasificación supervisada de textos**. Permite cargar conjuntos de datos etiquetados en formato CSV, inspeccionar su contenido, elegir métodos de extracción de características y clasificadores, entrenar modelos, revisar sus métricas y utilizar los modelos entrenados para clasificar textos nuevos.

La aplicación está diseñada para trabajar con diferentes conjuntos de datos y etiquetas, sin depender de un conjunto fijo de clases.

## Enlaces del proyecto

- **Repositorio de GitHub:** [https://github.com/Brayan-xd/textlab_web](https://github.com/Brayan-xd/textlab_web)
- **Carpeta de evidencias:** [https://github.com/Brayan-xd/textlab_web/tree/main/evidencias](https://github.com/Brayan-xd/textlab_web/tree/main/evidencias)
- **Aplicación Web:** https://textlabweb-production.up.railway.app/

## Funcionalidades principales

- Carga de archivos CSV y selección de las columnas de texto y etiqueta.
- Lectura de archivos con codificaciones UTF-8, UTF-8 con BOM y Latin-1.
- Inspección de valores faltantes, registros duplicados y distribución de clases.
- Selección de características lingüísticas: n-gramas, etiquetas POS, stopwords y content words.
- Representación de texto mediante frecuencia, presencia binaria o TF-IDF.
- Entrenamiento con Regresión Logística, SVM, Random Forest y Naive Bayes.
- División de datos para entrenamiento y prueba, y validación cruzada estratificada.
- Evaluación con accuracy, precision macro, recall macro, F1-score macro, métricas por clase y matriz de confusión.
- Registro y comparación de experimentos durante la sesión.
- Predicción de textos individuales y predicción por lotes a partir de un CSV sin etiquetas.

## Consideraciones sobre el volumen de datos

TextLab está diseñado para trabajar con diferentes conjuntos de datos de clasificación de textos, sin depender de una cantidad fija de registros o de clases predeterminadas. Sin embargo, es importante considerar que el rendimiento de la aplicación depende de los recursos disponibles en el equipo o servidor donde se ejecuta.

Actualmente, el sistema procesa los archivos CSV utilizando memoria RAM y realiza el entrenamiento de los modelos mediante procesamiento local en Python. Por esta razón, cuando se utilizan conjuntos de datos demasiado grandes, la carga, inspección y entrenamiento pueden volverse lentos o incluso superar la capacidad de memoria y procesamiento del servidor donde está desplegada la aplicación.

Para las pruebas con conjuntos de datos de mayor tamaño, se recomienda ejecutar TextLab en un entorno local que cuente con recursos suficientes de memoria RAM y procesamiento. Esto permite aprovechar los recursos disponibles del equipo en lugar de depender de las limitaciones del servidor de despliegue.

**Nota:** No existe un límite fijo de registros establecido en el código, por lo que la cantidad de datos que puede procesarse dependerá de las características del conjunto de datos y de los recursos disponibles en el entorno de ejecución.

## Tecnologías utilizadas

### Backend

- **Python:** lenguaje principal del servidor.
- **FastAPI:** creación de la API REST.
- **Pandas y NumPy:** lectura y manipulación de los conjuntos de datos.
- **Scikit-learn:** preprocesamiento, vectorización, entrenamiento, validación y evaluación de modelos.
- **NLTK:** recursos lingüísticos, stopwords y etiquetado POS.
- **Uvicorn:** servidor ASGI para ejecutar la aplicación.

### Frontend

- **HTML, CSS y JavaScript:** estructura, estilos e interacción de la interfaz.
- **Tailwind CSS:** estilos visuales.
- **Chart.js:** gráficas de evaluación.
- **Google Fonts:** tipografías Plus Jakarta Sans e IBM Plex Mono.

Tailwind CSS, Chart.js y las tipografías se cargan desde servicios externos, por lo que el navegador necesita conexión a Internet para obtener esos recursos.

## Requisitos

- Python 3.10 o superior.
- `pip` para instalar las dependencias.
- Navegador web actualizado.
- Conexión a Internet para cargar los recursos externos del frontend y, cuando sea necesario, descargar los recursos lingüísticos de NLTK.

## Instalación y ejecución local

1. Clona el repositorio:

   ```bash
   git clone https://github.com/Brayan-xd/textlab_web.git
   ```

2. Entra en la carpeta del proyecto:

   ```bash
   cd textlab_web
   ```

3. Crea un entorno virtual (recomendado):

   **Windows**

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   **Linux o macOS**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```
   

   O de no funcionar


   ```bash
   pip install fastapi "uvicorn[standard]" python-multipart numpy pandas nltk scikit-learn
   ```

4. Ejecuta el servidor desde la carpeta donde se encuentran `main.py` e `index.html`:

   ```bash
   uvicorn main:app --reload
   ```

5. Abre la aplicación en el navegador:

   **http://127.0.0.1:8000**

La documentación interactiva de la API de FastAPI está disponible en:

**http://127.0.0.1:8000/docs**

Al iniciar, el backend comprueba si están disponibles algunos recursos de NLTK y trata de descargarlos si hacen falta.

## Uso de la aplicación

### 1. Workbench

Esta pestaña concentra la preparación del conjunto de datos y el entrenamiento del modelo.

1. Presiona **Cargar CSV** y selecciona un archivo que contenga textos y sus etiquetas.
2. Selecciona la columna que contiene los textos y la columna que contiene las clases.
3. Usa **Filtrar nulos** y **Quitar duplicados** para definir la limpieza que se muestra al inspeccionar el conjunto.
4. Presiona **Inspeccionar dataset** para consultar el número de muestras útiles, las clases identificadas, los valores faltantes, los duplicados y la distribución de las clases.
5. En **Extracción de características**, selecciona el tipo de característica, el orden y alcance de los n-gramas, y la representación del texto.
6. En **Configuración y entrenamiento**, elige el clasificador, la proporción Train/Test, la validación cruzada y los hiperparámetros disponibles.
7. Presiona **Ejecutar entrenamiento**. Al terminar, la aplicación muestra los resultados en la pestaña **Evaluación**.

### 2. Evaluación

Presenta los resultados del último experimento o del experimento seleccionado en el historial. Incluye accuracy, precision macro, recall macro, F1-score macro, una gráfica por clase, la matriz de confusión y, cuando se haya activado, los resultados de validación cruzada con su promedio y desviación estándar.

### 3. Comparador & Predicción

En esta pestaña se encuentra el historial de experimentos realizados durante la sesión. Al seleccionar una fila se puede volver a consultar la evaluación correspondiente.

Para realizar predicciones:

- **Texto individual:** selecciona un modelo entrenado, escribe el texto y presiona **Predecir texto**.
- **CSV por lotes:** selecciona un modelo, escribe el nombre exacto de la columna que contiene los textos, carga un CSV sin etiquetas y presiona **Predecir CSV**. La interfaz muestra hasta 50 filas de resultados.

## API REST

Los endpoints principales implementados en `main.py` son los siguientes:

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Entrega la interfaz web (`index.html`). |
| `GET` | `/favicon.ico` | Responde con estado 204, sin contenido. |
| `POST` | `/api/upload-csv` | Recibe un CSV, lo carga en memoria y devuelve columnas, número de filas y una vista previa. |
| `POST` | `/api/inspect-dataset` | Inspecciona nulos y duplicados, aplica los filtros solicitados para el resumen y calcula la distribución de clases. |
| `POST` | `/api/train` | Construye el pipeline, entrena y evalúa un clasificador, y guarda el experimento en memoria. |
| `POST` | `/api/predict-text` | Predice la clase de un texto usando un experimento entrenado. |
| `POST` | `/api/predict-csv` | Predice las clases de los textos de una columna de un CSV. |

### Datos de entrada de los endpoints

**`POST /api/upload-csv`** recibe un archivo mediante `multipart/form-data`, con el campo `file`.

**`POST /api/inspect-dataset`** recibe un objeto JSON. Ejemplo:

```json
{
  "text_col": "text",
  "label_col": "label",
  "drop_duplicates": true,
  "drop_na": true
}
```

**`POST /api/train`** recibe un objeto JSON con la configuración del experimento. Ejemplo:

```json
{
  "text_col": "text",
  "label_col": "label",
  "feature_type": "ngrams",
  "ngram_size": 2,
  "ngram_scope": "exact",
  "representation": "tfidf",
  "classifier": "logreg",
  "test_size": 0.2,
  "cv_folds": 5,
  "params": {
    "penalty": "l2",
    "C": 1.0
  }
}
```

Valores contemplados por el backend:

- `feature_type`: `ngrams`, `pos`, `stopwords` o `content_words`.
- `ngram_size`: `1`, `2` o `3`.
- `ngram_scope`: `exact` o `up_to`.
- `representation`: `tfidf`, `freq` o `binary`.
- `classifier`: `logreg`, `svm`, `rf` o `nb`.
- `test_size`: proporción reservada para prueba, por ejemplo `0.2`.
- `cv_folds`: `0`, `3`, `5` o `10`.
- `params`: hiperparámetros correspondientes al clasificador elegido.

**`POST /api/predict-text`** recibe JSON:

```json
{
  "experiment_id": "EXP-01",
  "text": "El desempeño de la empresa mejoró durante el periodo."
}
```

**`POST /api/predict-csv`** recibe el archivo en `multipart/form-data` con el campo `file`, y los parámetros `experiment_id` y `text_col` en la URL. Ejemplo conceptual:

```text
/api/predict-csv?experiment_id=EXP-01&text_col=text
```

## Organización de evidencias

Las evidencias del desarrollo se encuentran en la carpeta `evidencias` del repositorio. Esta carpeta reúne los registros de los cuatro momentos del proyecto:

1. **Momento 1 — Inicio:** creación del repositorio y tablero, configuración inicial del backend y diseño de la interfaz.
2. **Momento 2 — Primer avance:** carga e inspección de datasets, limpieza, selección de columnas y preprocesamiento lingüístico.
3. **Momento 3 — Integración:** entrenamiento, evaluación, comparación de experimentos, predicción y conexión entre frontend y backend.
4. **Momento 4 — Cierre:** pruebas finales, validación con conjuntos de datos externos, revisión de la aplicación desplegada y consolidación de entregables.

Consulta las capturas y registros disponibles en [la carpeta de evidencias](https://github.com/Brayan-xd/textlab_web/tree/main/evidencias).

## Consideraciones de implementación

- El conjunto de datos cargado y los experimentos se guardan en variables de memoria del servidor. Al reiniciar el proceso, la información de la sesión se pierde.
- El estado se comparte en el proceso del servidor y no está aislado por usuario. La implementación actual está orientada a una sesión de trabajo y no a múltiples usuarios concurrentes.
- El endpoint de entrenamiento elimina valores faltantes y duplicados de las columnas seleccionadas antes de crear las particiones.
- La validación cruzada estratificada se ejecuta sobre los datos de entrenamiento. El pipeline incluye el preprocesador y el vectorizador para que se ajusten dentro de cada partición.
- La configuración CORS actual permite cualquier origen. Para un despliegue de producción debe limitarse a los dominios autorizados.
- En `index.html`, la interfaz envía el campo `ngram_mode`, mientras que el backend recibe `ngram_scope`. Para que el selector de alcance de n-gramas se aplique, ambos nombres deben coincidir.
- El backend devuelve métricas macro, pero no incluye `f1_weighted`, aunque la interfaz contempla mostrar ese dato. Por ello, ese valor puede aparecer como `--`.
- El clasificador SVM se crea con `probability=False`. En consecuencia, la respuesta de predicción utiliza un valor de respaldo de 100 % y no una probabilidad calibrada para esa clase.

## Estado del despliegue

La ejecución local se realiza en `http://127.0.0.1:8000`. No se incluye una URL pública de despliegue porque no fue proporcionada en la información disponible. Cuando exista una dirección pública, puede agregarse en esta sección.

---

**Proyecto:** TextLab — Plataforma web para la clasificación de textos  
**Repositorio:** [Brayan-xd/textlab_web](https://github.com/Brayan-xd/textlab_web)
