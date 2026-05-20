# 🎭 Musical Lyric Analysis Repository

Un pipeline de Ciencia de Datos diseñado para el scraping, procesamiento de lenguaje natural (NLP) y análisis de sentimientos de letras de musicales de Broadway. Este repositorio permite construir un dataset robusto que incluye métricas léxicas, metadatos históricos de Wikipedia y clasificaciones emocionales granulares.

## 🚀 Características

* **Scraping Automatizado**: Extracción de letras desde `allmusicals.com` organizada por orden alfabético y musicales individuales.
* **Enriquecimiento de Metadatos**: Integración con la API de Wikipedia para obtener automáticamente el año de estreno, compositor y letrista.
* **Análisis Léxico**: Cálculo de riqueza léxica, densidad de contenido y extracción de palabras clave utilizando `spaCy`.
* **Análisis de Sentimientos Multimodal**:
    * **Modelo Base**: Clasificación en 7 emociones (Hartmann).
    * **GoEmotions**: Clasificación granular en 28 categorías emocionales mediante `RoBERTa`.
* **Interfaz de Terminal (TUI)**: Feedback visual avanzado en consola utilizando la librería `rich`.

## 📁 Estructura del Proyecto

```text
.
├── data/                   # Archivos .txt de letras organizados por Musical/Acto
├── scrape.py               # Script principal de extracción y análisis base
├── wiki_patch.py           # Script para parchear metadatos desde Wikipedia
├── go_emotions.py          # Enriquecimiento con el modelo GoEmotions (28 clases)
├── main.py                 # Punto de entrada principal
├── musicals_dataset.csv    # Dataset inicial generado
└── full_data.csv           # Dataset final enriquecido
````
## 🛠️ Requisitos Técnicos
El proyecto está desarrollado en Python 3.13+ y utiliza las siguientes librerías:

Procesamiento de Datos: pandas, numpy

NLP & Modelos: spacy, transformers (HuggingFace), torch

Web Scraping: beautifulsoup4, requests, wikipedia

Instalación de modelos de lenguaje
Es necesario descargar el modelo de spaCy para el análisis léxico:

```bash
python -m spacy download en_core_web_sm
```
## 📖 Guía de Uso
El flujo de trabajo está diseñado para ejecutarse en etapas para garantizar la integridad de los datos:

1. Extracción Inicial
Ejecuta scrape.py para descargar las letras y generar las métricas de j-hartmann/emotion-english-distilroberta-base.

```bash
python scrape.py
```

2. Enriquecimiento de Metadatos
Usa wiki_patch.py para buscar información histórica de los musicales en Wikipedia. El script cuenta con un TEST_MODE para validar resultados antes del procesamiento masivo.

```bash
python wiki_patch.py
```

3. Análisis Emocional Profundo
Finalmente, corre go_emotions.py. Este script aplica el modelo SamLowe/roberta-base-go_emotions, manejando el truncamiento de texto y chunking para procesar canciones largas sin perder contexto.

```
python go_emotions.py
```

## 📊 Dataset Resultante
El archivo go_emotions_data.csv final contiene:

Metadatos: Título, Musical, Acto (Subsection), Año, Compositor, Letrista.

Métricas Léxicas: Riqueza léxica, densidad léxica y top 5 palabras más usadas.

Emociones (28 columnas): Scores de confianza para categorías como joy, sadness, admiration, remorse, etc.
