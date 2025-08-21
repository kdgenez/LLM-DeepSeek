import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, date
import io
import base64

# Para el LLM - usando DeepSeek
import requests
import json
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from typing import Optional, List, Any

# Configuración de la página
st.set_page_config(
    page_title="Plataforma de IA Integrada",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #00cc88;
    }
    .assistant-message {
        background-color: #f8f9fa;
        border-left-color: #1f77b4;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #00cc88;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🤖 Plataforma de IA Integrada</h1>', unsafe_allow_html=True)
st.markdown("### 📊 Análisis de Datos + 🧠 Asistente Inteligente")

# Clase personalizada para DeepSeek
class DeepSeekLLM(LLM):
    api_key: str
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 1000
    base_url: str = "https://api.deepseek.com/v1/chat/completions"
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            return f"Error de conexión con DeepSeek: {str(e)}"
        except KeyError as e:
            return f"Error en respuesta de DeepSeek: {str(e)}"
        except Exception as e:
            return f"Error inesperado: {str(e)}"
    
    @property
    def _identifying_params(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

# Configuración de APIs (usando st.secrets)
@st.cache_resource
def initialize_llm():
    try:
        # Cambiamos a DeepSeek
        deepseek_api_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        if not deepseek_api_key:
            st.warning("⚠️ No se encontró DEEPSEEK_API_KEY en los secrets. Usando modo demo.")
            return None
        
        llm = DeepSeekLLM(
            api_key=deepseek_api_key,
            model="deepseek-chat",  # Modelo principal de DeepSeek
            temperature=0.7,
            max_tokens=1000
        )
        return llm
    except Exception as e:
        st.error(f"Error inicializando LLM: {e}")
        return None

# Inicializar memoria de conversación
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selector de tema
    theme = st.selectbox(
        "🎨 Tema de gráficos",
        ["plotly", "seaborn", "ggplot2", "simple_white"],
        index=0
    )
    
    # Configuraciones del asistente
    st.subheader("🤖 Configuración del Asistente")
    assistant_mode = st.selectbox(
        "Modo del asistente",
        ["General", "Análisis de Datos", "Agricultura", "Programación", "Negocios"],
        index=0
    )
    
    temperature = st.slider("🌡️ Creatividad", 0.0, 1.0, 0.7, 0.1)
    
    st.subheader("📊 Datos Cargados")
    if st.session_state.get('df') is not None:
        st.success(f"✅ Dataset cargado: {st.session_state.df.shape[0]} filas, {st.session_state.df.shape[1]} columnas")
    else:
        st.info("📁 No hay dataset cargado")

# Crear pestañas principales
tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis de Datos", "🤖 Asistente IA", "📈 Dashboard", "⚙️ Configuración"])

# TAB 1: ANÁLISIS DE DATOS
with tab1:
    st.header("📊 Análisis Exploratorio de Datos")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Carga de archivos
        uploaded_file = st.file_uploader(
            "📁 Cargar archivo de datos",
            type=['csv', 'xlsx', 'json'],
            help="Formatos soportados: CSV, Excel, JSON"
        )
        
        if uploaded_file is not None:
            try:
                # Leer archivo según tipo
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                st.session_state.df = df
                st.success(f"✅ Archivo cargado: {df.shape[0]} filas y {df.shape[1]} columnas")
                
            except Exception as e:
                st.error(f"❌ Error al cargar el archivo: {e}")
    
    with col2:
        # Dataset de ejemplo
        if st.button("📊 Usar Dataset de Ejemplo"):
            # Crear dataset sintético
            np.random.seed(42)
            n_samples = 1000
            
            df = pd.DataFrame({
                'fecha': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
                'ventas': np.random.normal(5000, 1500, n_samples).astype(int),
                'categoria': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
                'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], n_samples),
                'precio': np.random.uniform(10, 100, n_samples).round(2),
                'cantidad': np.random.poisson(20, n_samples),
                'satisfaccion': np.random.uniform(1, 10, n_samples).round(1)
            })
            
            st.session_state.df = df
            st.success("✅ Dataset de ejemplo cargado")
    
    # Análisis del dataset cargado
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        
        st.subheader("🔍 Vista General del Dataset")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Filas", f"{df.shape[0]:,}")
        with col2:
            st.metric("📋 Columnas", f"{df.shape[1]:,}")
        with col3:
            st.metric("💾 Memoria", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        with col4:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
            st.metric("❓ Datos Faltantes", f"{missing_pct:.1f}%")
        
        # Mostrar datos
        if st.checkbox("👀 Mostrar datos"):
            st.dataframe(df.head(100), use_container_width=True)
        
        # Información del dataset
        if st.checkbox("ℹ️ Información del dataset"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        # Estadísticas descriptivas
        if st.checkbox("📈 Estadísticas descriptivas"):
            st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader("📊 Visualizaciones")
        
        # Selector de tipo de gráfico
        viz_type = st.selectbox(
            "Tipo de visualización",
            ["Histograma", "Scatter Plot", "Box Plot", "Correlación", "Distribución", "Series Temporales"]
        )
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        col1, col2 = st.columns(2)
        
        if viz_type == "Histograma" and numeric_cols:
            with col1:
                selected_col = st.selectbox("Columna", numeric_cols)
            with col2:
                bins = st.slider("Número de bins", 10, 100, 30)
            
            fig = px.histogram(df, x=selected_col, nbins=bins, template=theme)
            fig.update_layout(title=f"Distribución de {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
            with col1:
                x_col = st.selectbox("Eje X", numeric_cols)
            with col2:
                y_col = st.selectbox("Eje Y", [col for col in numeric_cols if col != x_col])
            
            color_col = None
            if categorical_cols:
                color_col = st.selectbox("Color (opcional)", [None] + categorical_cols)
            
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, template=theme)
            fig.update_layout(title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot" and numeric_cols:
            selected_col = st.selectbox("Columna numérica", numeric_cols)
            group_col = None
            if categorical_cols:
                group_col = st.selectbox("Agrupar por (opcional)", [None] + categorical_cols)
            
            fig = px.box(df, y=selected_col, x=group_col, template=theme)
            fig.update_layout(title=f"Box Plot de {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Correlación" and len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", template=theme)
            fig.update_layout(title="Matriz de Correlación")
            st.plotly_chart(fig, use_container_width=True)
        
        # Guardar resultados del análisis
        st.session_state.analysis_results = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'missing_data': df.isnull().sum().to_dict()
        }

# TAB 2: ASISTENTE IA
with tab2:
    st.header("🤖 Asistente de IA Conversacional")
    
    llm = initialize_llm()
    
    if llm is None:
        st.warning("⚠️ No se pudo inicializar el modelo de IA. Verifica la configuración de API keys.")
        st.info("💡 Para usar esta funcionalidad, agrega tu DEEPSEEK_API_KEY en los secrets de Streamlit.")
        st.info("🔗 Obtén tu API key en: https://platform.deepseek.com/api_keys")
    else:
        # Sistema de prompts según el modo
        system_prompts = {
            "General": "Eres un asistente útil y conocedor. Responde de manera clara y precisa.",
            "Análisis de Datos": "Eres un experto en análisis de datos y ciencia de datos. Ayuda con interpretación de datos, estadísticas y visualizaciones.",
            "Agricultura": "Eres un experto en agricultura. Ayuda con cultivos, fertilización, enfermedades de plantas y buenas prácticas agrícolas.",
            "Programación": "Eres un experto programador. Ayuda con código, debugging y mejores prácticas de desarrollo.",
            "Negocios": "Eres un consultor de negocios experto. Ayuda con estrategias, análisis de mercado y decisiones empresariales."
        }
        
        # Contexto adicional si hay datos cargados
        data_context = ""
        if 'df' in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
            data_context = f"""
            
CONTEXTO DE DATOS CARGADOS:
- Dataset con {df.shape[0]} filas y {df.shape[1]} columnas
- Columnas numéricas: {', '.join(st.session_state.analysis_results.get('numeric_cols', []))}
- Columnas categóricas: {', '.join(st.session_state.analysis_results.get('categorical_cols', []))}
- Datos faltantes: {sum(st.session_state.analysis_results.get('missing_data', {}).values())} valores
            """
        
        full_prompt = system_prompts[assistant_mode] + data_context
        
        # Interfaz de chat
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_area(
                "💬 Escribe tu pregunta:",
                height=100,
                placeholder="Pregunta lo que necesites..."
            )
        
        with col2:
            st.write("**Ejemplos de preguntas:**")
            if assistant_mode == "Análisis de Datos" and 'df' in st.session_state:
                examples = [
                    "Explica los datos que tengo cargados",
                    "¿Qué patrones ves en los datos?",
                    "Recomienda visualizaciones",
                    "Identifica valores atípicos"
                ]
            elif assistant_mode == "Agricultura":
                examples = [
                    "¿Cómo fertilizar maíz?",
                    "Control de plagas en tomate",
                    "Mejores cultivos para suelos ácidos",
                    "Calendario de siembra"
                ]
            else:
                examples = [
                    "Explícame un concepto",
                    "Ayúdame con un problema",
                    "Dame recomendaciones",
                    "Analiza esta situación"
                ]
            
            for example in examples:
                if st.button(f"💡 {example}", key=example):
                    user_input = example
        
        # Botones de acción
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            send_message = st.button("📤 Enviar", type="primary")
        with col2:
            clear_chat = st.button("🗑️ Limpiar Chat")
        with col3:
            if st.button("💾 Guardar Chat"):
                chat_text = "\n\n".join([f"Usuario: {msg['content']}" if msg['type'] == 'user' 
                                       else f"Asistente: {msg['content']}" 
                                       for msg in st.session_state.chat_history])
                st.download_button(
                    "⬇️ Descargar",
                    chat_text,
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
        
        if clear_chat:
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()
        
        if send_message and user_input.strip():
            try:
                # Agregar mensaje del usuario al historial
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': user_input,
                    'timestamp': datetime.now()
                })
                
                # Crear prompt con contexto
                template = f"""
                {full_prompt}
                
                Historial de conversación:
                {{history}}
                
                Usuario: {{input}}
                Asistente:
                """
                
                prompt = PromptTemplate(
                    input_variables=["history", "input"],
                    template=template
                )
                
                # Crear cadena con memoria
                chain = LLMChain(
                    llm=llm,
                    prompt=prompt,
                    memory=st.session_state.memory,
                    verbose=False
                )
                
                # Generar respuesta usando DeepSeek
                with st.spinner("🤔 Pensando con DeepSeek..."):
                    # Construir el prompt completo manualmente para DeepSeek
                    conversation_history = ""
                    for msg in st.session_state.chat_history[-5:]:  # Últimos 5 mensajes para contexto
                        role = "Usuario" if msg['type'] == 'user' else "Asistente"
                        conversation_history += f"{role}: {msg['content']}\n"
                    
                    full_prompt = f"{full_prompt}\n\nHistorial reciente:\n{conversation_history}\n\nUsuario: {user_input}\nAsistente:"
                    
                    response = llm(full_prompt)
                
                # Agregar respuesta al historial
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': response,
                    'timestamp': datetime.now()
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generando respuesta: {e}")
        
        # Mostrar historial de chat
        if st.session_state.chat_history:
            st.subheader("💬 Conversación")
            
            for i, message in enumerate(reversed(st.session_state.chat_history[-10:])):  # Mostrar últimos 10
                if message['type'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>🙋 Tú:</strong><br>
                        {message['content']}<br>
                        <small>{message['timestamp'].strftime('%H:%M:%S')}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>🤖 Asistente:</strong><br>
                        {message['content']}<br>
                        <small>{message['timestamp'].strftime('%H:%M:%S')}</small>
                    </div>
                    """, unsafe_allow_html=True)

# TAB 3: DASHBOARD
with tab3:
    st.header("📈 Dashboard Interactivo")
    
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        
        # Métricas clave
        st.subheader("📊 KPIs Principales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 1:
            with col1:
                st.metric(
                    f"📈 Total {numeric_cols[0]}", 
                    f"{df[numeric_cols[0]].sum():,.0f}",
                    delta=f"{df[numeric_cols[0]].std():.1f} σ"
                )
        
        if len(numeric_cols) >= 2:
            with col2:
                st.metric(
                    f"📊 Promedio {numeric_cols[1]}", 
                    f"{df[numeric_cols[1]].mean():.2f}",
                    delta=f"{df[numeric_cols[1]].median():.2f} mediana"
                )
        
        with col3:
            st.metric("📋 Registros", f"{len(df):,}")
        
        with col4:
            st.metric("🔄 Completitud", f"{(1 - df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.1f}%")
        
        # Gráficos del dashboard
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de líneas/barras
                chart_type = st.selectbox("Tipo de gráfico", ["line", "bar", "area"])
                x_col = st.selectbox("Eje X", df.columns.tolist(), key="dash_x")
                y_col = st.selectbox("Eje Y", numeric_cols, key="dash_y")
                
                if chart_type == "line":
                    fig = px.line(df.head(100), x=x_col, y=y_col, template=theme)
                elif chart_type == "bar":
                    fig = px.bar(df.head(20), x=x_col, y=y_col, template=theme)
                else:
                    fig = px.area(df.head(100), x=x_col, y=y_col, template=theme)
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Gráfico de distribución
                dist_col = st.selectbox("Columna para distribución", numeric_cols, key="dist")
                fig = px.histogram(df, x=dist_col, template=theme)
                fig.update_layout(title=f"Distribución de {dist_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        # Tabla resumen
        st.subheader("📋 Resumen de Datos")
        summary_type = st.selectbox("Tipo de resumen", ["Estadísticas", "Valores únicos", "Datos faltantes"])
        
        if summary_type == "Estadísticas":
            st.dataframe(df.describe(), use_container_width=True)
        elif summary_type == "Valores únicos":
            unique_counts = pd.DataFrame({
                'Columna': df.columns,
                'Valores únicos': [df[col].nunique() for col in df.columns],
                'Tipo': [str(df[col].dtype) for col in df.columns]
            })
            st.dataframe(unique_counts, use_container_width=True)
        else:
            missing_data = pd.DataFrame({
                'Columna': df.columns,
                'Valores faltantes': df.isnull().sum(),
                'Porcentaje': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(missing_data, use_container_width=True)
    
    else:
        st.info("📁 Carga un dataset en la pestaña 'Análisis de Datos' para ver el dashboard")

# TAB 4: CONFIGURACIÓN
with tab4:
    st.header("⚙️ Configuración de la Aplicación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔑 APIs y Tokens")
        
        # Verificar APIs
        apis_status = {}
        
        # DeepSeek
        deepseek_key = st.secrets.get("DEEPSEEK_API_KEY", "")
        apis_status["DeepSeek"] = "✅ Configurada" if deepseek_key else "❌ No configurada"
        
        for api, status in apis_status.items():
            st.write(f"**{api}:** {status}")
        
        st.info("""
        **Para configurar las APIs:**
        1. Ve a la configuración de tu app en Streamlit Cloud
        2. En la sección 'Secrets', agrega:
        ```
        DEEPSEEK_API_KEY = "tu_api_key_aqui"
        ```
        3. Obtén tu API key en: https://platform.deepseek.com/api_keys
        """)
    
    with col2:
        st.subheader("📊 Estado de la Aplicación")
        
        # Información del sistema
        st.write(f"**Memoria de chat:** {len(st.session_state.chat_history)} mensajes")
        st.write(f"**Dataset cargado:** {'Sí' if 'df' in st.session_state else 'No'}")
        if 'df' in st.session_state:
            st.write(f"**Tamaño del dataset:** {st.session_state.df.shape}")
        
        # Limpiar caché
        if st.button("🗑️ Limpiar todo el caché"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Caché limpiado")
        
        # Reset aplicación
        if st.button("🔄 Reiniciar aplicación"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    st.subheader("📚 Documentación y Ayuda")
    
    with st.expander("🚀 Cómo usar la aplicación"):
        st.markdown("""
        ### 📊 Análisis de Datos
        1. **Cargar datos:** Sube un archivo CSV, Excel o JSON
        2. **Explorar:** Revisa estadísticas y visualizaciones
        3. **Analizar:** Usa diferentes tipos de gráficos
        
        ### 🤖 Asistente IA
        1. **Configurar modo:** Selecciona el tipo de asistente
        2. **Conversar:** Haz preguntas en lenguaje natural
        3. **Contexto:** El asistente conoce tus datos cargados
        
        ### 📈 Dashboard
        - **KPIs automáticos** de tus datos
        - **Gráficos interactivos** personalizables
        - **Resúmenes** en diferentes formatos
        
        ### ⚙️ Configuración
        - **APIs:** Configura tus tokens de OpenAI
        - **Estado:** Monitorea el uso de memoria
        - **Mantenimiento:** Limpia caché cuando sea necesario
        """)
    
    with st.expander("🔧 Solución de problemas"):
        st.markdown("""
        **Problemas comunes:**
        
        1. **"No se pudo inicializar el LLM"**
           - Verifica que OPENAI_API_KEY esté en los secrets
           - Asegúrate de que la API key sea válida
        
        2. **"Error al cargar archivo"**
           - Verifica el formato del archivo (CSV, Excel, JSON)
           - Asegúrate de que el archivo no esté corrupto
        
        3. **"La aplicación va lenta"**
           - Usa el botón "Limpiar caché" en configuración
           - Reinicia la aplicación si es necesario
        
        4. **"No veo mis datos en el dashboard"**
           - Primero carga datos en la pestaña "Análisis de Datos"
           - Verifica que el archivo se haya cargado correctamente
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🚀 <strong>Plataforma de IA Integrada</strong> | 
    Desarrollado para el curso de Introducción a la IA | 
    Powered by DeepSeek AI & Streamlit
</div>
""", unsafe_allow_html=True)
