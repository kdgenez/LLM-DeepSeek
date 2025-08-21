# 🤖 Plataforma de IA Integrada

Una aplicación completa que combina **análisis de datos avanzado** y **asistente de IA conversacional** en una sola plataforma, desarrollada con Streamlit y potenciada por **DeepSeek AI**.

## 🚀 Características Principales

### 📊 **Análisis de Datos**
- ✅ Carga de archivos CSV, Excel y JSON
- ✅ Análisis exploratorio automático
- ✅ Visualizaciones interactivas con Plotly
- ✅ Estadísticas descriptivas completas
- ✅ Detección de valores faltantes y atípicos
- ✅ Múltiples tipos de gráficos (histogramas, scatter plots, box plots, correlaciones)

### 🤖 **Asistente de IA Conversacional**
- ✅ Powered by **DeepSeek AI** (modelo avanzado y económico)
- ✅ Múltiples modos especializados:
  - 🔢 Análisis de Datos
  - 🌱 Agricultura (modo original)
  - 💻 Programación
  - 📈 Negocios
  - 🎯 General
- ✅ Memoria conversacional persistente
- ✅ Contexto automático de datos cargados
- ✅ Exportación de conversaciones

### 📈 **Dashboard Interactivo**
- ✅ KPIs automáticos de tus datos
- ✅ Visualizaciones dinámicas
- ✅ Resúmenes configurables
- ✅ Métricas en tiempo real

### ⚙️ **Configuración Avanzada**
- ✅ Temas personalizables para gráficos
- ✅ Control de creatividad del AI
- ✅ Gestión de memoria y caché
- ✅ Diagnósticos del sistema

## 🛠️ Instalación y Configuración

### 1. **Preparar el Repositorio**
```bash
git clone tu-repositorio
cd tu-repositorio
```

### 2. **Configurar Dependencias**
Asegúrate de que tu `requirements.txt` esté actualizado con las nuevas dependencias.

### 3. **Configurar API Keys**
En **Streamlit Cloud**, ve a la configuración de tu app y agrega en **Secrets**:

```toml
DEEPSEEK_API_KEY = "tu-api-key-de-deepseek-aqui"
```

#### 🔑 **Cómo obtener tu DeepSeek API Key:**
1. Ve a [DeepSeek Platform](https://platform.deepseek.com/)
2. Crea una cuenta o inicia sesión
3. Ve a [API Keys](https://platform.deepseek.com/api_keys)
4. Genera una nueva API key
5. Cópiala y pégala en los secrets de Streamlit

#### 💰 **Ventajas de DeepSeek:**
- 🚀 **Rendimiento:** Comparable a GPT-4
- 💰 **Económico:** Hasta 10x más barato que OpenAI
- 🌍 **Multilingüe:** Excelente soporte en español
- ⚡ **Rápido:** Respuestas en segundos
- 🔐 **Confiable:** Infraestructura empresarial

### 4. **Desplegar en Streamlit Cloud**
1. Conecta tu repositorio a [Streamlit Cloud](https://streamlit.io/cloud)
2. Selecciona el archivo `app.py`
3. Configura los secrets con tu OpenAI API Key
4. ¡Deploy!

## 📖 Guía de Uso

### 📊 **Análisis de Datos**
1. **Cargar Datos:**
   - Haz clic en "📁 Cargar archivo de datos"
   - Soporta CSV, Excel (.xlsx, .xls) y JSON
   - O usa "📊 Usar Dataset de Ejemplo" para probar

2. **Explorar:**
   - Revisa métricas básicas (filas, columnas, memoria)
   - Visualiza los primeros registros
   - Examina estadísticas descriptivas

3. **Visualizar:**
   - Selecciona diferentes tipos de gráficos
   - Personaliza ejes y colores
   - Explora correlaciones y distribuciones

### 🤖 **Asistente de IA**
1. **Configurar Modo:**
   - Selecciona el modo en la barra lateral
   - Ajusta la creatividad (temperatura)

2. **Conversar:**
   - Escribe tu pregunta en el área de texto
   - O usa los ejemplos sugeridos
   - El asistente tiene contexto de tus datos cargados

3. **Gestionar:**
   - Limpia el chat cuando necesites
   - Guarda conversaciones importantes
   - Revisa el historial reciente

### 📈 **Dashboard**
- **KPIs automáticos** basados en tus datos
- **Gráficos interactivos** que puedes personalizar
- **Resúmenes** en diferentes formatos

### ⚙️ **Configuración**
- Verifica el estado de tus APIs
- Monitorea el uso de memoria
- Limpia caché cuando sea necesario

## 🎯 **Casos de Uso**

### 📊 **Para Análisis de Datos:**
- "Analiza las tendencias en mis datos de ventas"
- "¿Qué patrones ves en este dataset?"
- "Recomienda visualizaciones para estos datos"
- "Identifica valores atípicos"

### 🌱 **Para Agricultura (modo original):**
- "¿Cómo fertilizar maíz de manera orgánica?"
- "Control de plagas en cultivos de tomate"
- "Mejores prácticas para suelos ácidos"
- "Calendario de siembra para clima tropical"

### 💻 **Para Programación:**
- "Explica este algoritmo de machine learning"
- "¿Cómo optimizar este código Python?"
- "Mejores prácticas para APIs REST"

### 📈 **Para Negocios:**
- "Analiza esta estrategia de mercado"
- "¿Cómo interpretar estos KPIs?"
- "Recomienda mejoras en el proceso"

## 🔧 **Diferencias vs Versión Original**

| Aspecto | Versión Original | Nueva Versión |
|---------|------------------|---------------|
| **LLM** | Llama3 via Groq | **DeepSeek AI** (más potente y económico) |
| **Funcionalidades** | Solo chat agrícola | Chat + Análisis de datos + Dashboard |
| **Interfaz** | Página simple | Multi-tab con sidebar |
| **Datos** | No maneja datos | Carga CSV/Excel/JSON |
| **Visualizaciones** | Ninguna | Plotly interactivo |
| **Memoria** | Sin persistencia | Conversaciones persistentes |
| **Modos** | Solo agricultura | 5 modos especializados |
| **Exportación** | No disponible | Descarga chats y datos |

## 🚨 **Solución de Problemas**

### ❌ **"No se pudo inicializar el LLM"**
- ✅ Verifica que `DEEPSEEK_API_KEY` esté en los secrets
- ✅ Asegúrate de que la API key sea válida
- ✅ Revisa tu saldo en DeepSeek Platform

### ❌ **"Error al cargar archivo"**
- ✅ Verifica el formato (CSV, Excel, JSON)
- ✅ Asegúrate de que el archivo no esté corrupto
- ✅ Revisa que no sea demasiado grande (< 200MB)

### ❌ **"La aplicación va lenta"**
- ✅ Usa "🗑️ Limpiar todo el caché" en configuración
- ✅ Reinicia la aplicación si es necesario
- ✅ Reduce el tamaño del dataset

### ❌ **"No veo mis datos en el dashboard"**
- ✅ Primero carga datos en "📊 Análisis de Datos"
- ✅ Verifica que el archivo se haya cargado correctamente
- ✅ Asegúrate de que el dataset tenga columnas numéricas

## 💡 **Tips para Mejores Resultados**

### 📊 **Para Análisis de Datos:**
- Usa datasets con columnas bien nombradas
- Incluye diferentes tipos de datos (numéricos, categóricos, fechas)
- Limpia datos básicos antes de cargar

### 🤖 **Para el Asistente:**
- Sé específico en tus preguntas
- Usa el modo correcto para tu consulta
- Referencia columnas específicas de tus datos

### 🎨 **Para Visualizaciones:**
- Experimenta con diferentes temas
- Usa colores para destacar categorías
- Combina múltiples tipos de gráficos

## 🔄 **Próximas Mejoras**

- [ ] Soporte para más formatos de archivo (Parquet, SQLite)
- [ ] Integración con bases de datos
- [ ] Modelos de machine learning básicos
- [ ] Exportación de gráficos en PDF
- [ ] Múltiples datasets simultáneos
- [ ] Análisis predictivo automatizado

## 📄 **Licencia**

Este proyecto está desarrollado para fines educativos como parte del curso de Introducción a la IA.

## 🤝 **Contribuciones**

¡Las contribuciones son bienvenidas! Si encuentras bugs o tienes ideas para mejoras, no dudes en reportarlas.

---

**🚀 ¡Disfruta explorando tus datos con DeepSeek AI!** 🤖📊
