# Embeddings y Búsqueda Vectorial en ChromaDB

## Qué es un embedding

Un embedding es una representación numérica de un texto en un espacio vectorial de alta
dimensionalidad. Un modelo de embeddings transforma una cadena de texto en un vector de
números reales, por ejemplo de 1536 dimensiones. La propiedad fundamental es que textos con
significado parecido quedan cerca en ese espacio, medido con una métrica de distancia como
la similitud coseno o la distancia euclídea. Esta propiedad es lo que permite buscar por
significado y no por coincidencia literal de palabras.

## Modelos de embeddings y su elección

Existen dos familias principales. Los modelos propietarios accedidos por API, como
OpenAIEmbeddings con los modelos text-embedding-3-small y text-embedding-3-large, ofrecen
alta calidad y no requieren cómputo local. Los modelos de código abierto, como los basados
en HuggingFaceEmbeddings o sentence-transformers, corren localmente y no tienen costo por
llamada, a cambio de requerir memoria y potencia de CPU o GPU. La regla de oro es que la
elección del modelo de embeddings debe ser coherente entre indexación y consulta.

## El error crítico de embeddings no coincidentes

El error número uno en pipelines RAG es indexar los documentos con un modelo de embeddings y
consultar con otro distinto. Si se indexa con OpenAIEmbeddings, que produce vectores de 1536
dimensiones con una orientación semántica específica, y luego se consulta con
HuggingFaceEmbeddings, que produce vectores de 384 dimensiones con otra orientación, las
distancias calculadas no tienen ningún sentido. Los resultados serán aparentemente aleatorios
y el sistema parecerá roto aunque el código no arroje errores. La defensa es encapsular la
creación del modelo de embeddings en una única función o clase y usar esa misma instancia
tanto para la ingesta como para el retriever. Nunca debe haber dos objetos de embeddings
distintos en el mismo proyecto.

## Qué es una base de datos vectorial

Una base de datos vectorial almacena embeddings y permite búsquedas de similitud eficientes
sobre millones de vectores. Además del vector, guarda el texto original del documento y sus
metadatos. ChromaDB es una base vectorial ligera y embebible que corre en el mismo proceso
de Python y persiste sus datos en disco, ideal para proyectos educativos y para prototipos
que luego pueden migrar a servicios gestionados.

## Persistencia en ChromaDB

ChromaDB ofrece un modo persistente mediante la clase PersistentClient, que recibe una ruta
de carpeta local, por ejemplo ./vectorstore. Cuando se usa esta clase, todo lo que se agrega
a una colección queda escrito en disco y sobrevive al cierre del proceso. La clase Chroma de
langchain_chroma envuelve a este cliente y expone una interfaz de LangChain. La persistencia
es fundamental para no reindexar el corpus en cada ejecución, lo cual ahorraría tiempo y,
si los embeddings son de API, ahorraría dinero real.

## Verificar antes de reindexar

Un pipeline profesional verifica si la colección ya existe y contiene documentos antes de
volver a indexar. La forma habitual es abrir el cliente persistente, listar las colecciones,
comprobar si la colección objetivo ya tiene un conteo mayor que cero y, en ese caso, saltar
la indexación. Solo si la colección no existe o está vacía se ejecuta la ingesta completa.
Esta verificación convierte el script en idempotente: ejecutarlo dos veces produce el mismo
estado final sin duplicar datos ni gastar cómputo.

## Métricas de distancia

ChromaDB permite configurar la métrica de distancia de la colección. Las más comunes son
la distancia al cuadrado L2 y la similitud coseno. La similitud coseno es la habitual para
embeddings de texto normalizados y devuelve valores cercanos a 1 para pares muy similares.
Es importante saber cuál usa la colección porque los puntajes reportados tienen interpretación
distinta. Un puntaje de 0.1 en L2 y un puntaje de 0.1 en coseno no significan lo mismo.

## Búsqueda por similitud

La búsqueda por similitud toma el embedding de la consulta y encuentra los k vectores más
cercanos en la colección. El parámetro k, o top_k, controla cuántos fragmentos se recuperan.
Un valor entre 3 y 5 es un buen punto de partida para la mayoría de los casos: suficiente
para cubrir variaciones de redacción y suficientemente acotado para no saturar el contexto
del modelo de lenguaje.

## Metadatos y trazabilidad

Cada documento almacenado en ChromaDB puede llevar metadatos como source, file_path y
chunk_index. Estos metadatos viajan con el fragmento cuando se recupera y permiten mostrar al
usuario de qué archivo provino cada fragmento. En un sistema de preguntas y respuestas
serio, la trazabilidad de la fuente es tan importante como la respuesta misma, porque permite
auditar y verificar la afirmación del modelo.
