# Checklist de Producción y Errores Comunes en RAG

## Top_k y presupuesto de contexto

El parámetro top_k controla cuántos fragmentos se recuperan y se inyectan en el prompt. Un
valor muy bajo puede no cubrir la información necesaria. Un valor muy alto provoca dos
problemas: se acerca o supera el límite de la ventana de contexto del modelo, lo que produce
errores, y degrada la atención del modelo por el fenómeno Lost in the Middle. La
recomendación práctica es fijar top_k entre 3 y 5 fragmentos. Con chunks de 500 tokens, eso
significa entre 1500 y 2500 tokens de contexto, que es una carga manejable y enfocada.

## El error del contexto infinito

Muchos desarrolladores noveles intentan resolver la falta de precisión aumentando el número
de fragmentos inyectados. Pasan 20 o 50 fragmentos al modelo pensando que más información es
mejor. El resultado es el opuesto al buscado: el modelo se confunde, aparecen errores de
límite de tokens y la calidad de la respuesta cae. La solución no es inyectar más contexto
sino mejorar la calidad de la recuperación: mejor chunking, mejor modelo de embeddings y un
top_k acotado.

## Embeddings no coincidentes, el error número uno

Ya se mencionó pero merece repetirse porque es el error más frecuente y más difícil de
diagnosticar. Si la ingesta usa OpenAIEmbeddings y la consulta usa HuggingFaceEmbeddings, o
si se cambió el modelo de embeddings a mitad del proyecto sin reindexar, los vectores viven
en espacios distintos y las distancias pierden sentido. El sistema no falla con una
excepción, simplemente devuelve resultados sin relación con la pregunta. La defensa es una
única fuente de verdad para el modelo de embeddings, usada tanto en la ingesta como en el
retriever, y reindexar por completo cada vez que se cambie ese modelo.

## Falta de persistencia

Reindexar todo el corpus en cada ejecución es un desperdicio. Si los embeddings son de API,
cada reindexación cuesta dinero. La solución es usar el cliente persistente de ChromaDB y
verificar la existencia y el conteo de la colección antes de indexar. El script de ingesta
debe ser idempotente: detectar que la base ya existe y saltar el trabajo pesado.

## Manejo de variables de entorno y secretos

Las llaves de API nunca deben quedar escritas en el código ni subirse al repositorio. La
práctica correcta es leerlas de variables de entorno cargadas desde un archivo .env, que se
excluye del control de versiones mediante el archivo .gitignore. Se acompaña el repositorio
con un archivo .env.example que muestra qué variables se esperan, sin contener valores
reales. Así otra persona puede clonar el proyecto, copiar el ejemplo y completar sus propias
credenciales.

## La pregunta trampa

Una prueba de calidad esencial en un sistema RAG es la pregunta trampa: una pregunta cuya
respuesta no está en los documentos. Un sistema bien construido debe responder que no tiene
acceso a esa información, no inventar. Esta prueba verifica que el prompt de veracidad y la
baja temperatura están funcionando. Si el sistema responde inventando, aunque sea una vez,
ha fallado el criterio de groundedness.

## La pregunta en dominio

La contraparte de la pregunta trampa es la pregunta cuya respuesta sí está en los documentos.
Esta prueba verifica que la recuperación encuentra los fragmentos correctos y que el modelo
los usa. Es útil observar los fragmentos recuperados y sus puntajes de similitud, no solo la
respuesta final, porque permite diagnosticar si un fallo vino de la recuperación o de la
generación.

## Separar ingesta de consulta

Un buen diseño separa el proceso de ingesta, que corre ocasionalmente y es costoso, del
proceso de consulta, que corre muchas veces y debe ser rápido. La ingesta produce una base
vectorial persistente. La consulta solo abre esa base, busca y genera. Esta separación mejora
el rendimiento, reduce costos y hace el sistema más fácil de razonar y de depurar.

## Observabilidad mínima

Un pipeline de producción debería registrar, como mínimo, la pregunta recibida, los
fragmentos recuperados con sus fuentes y puntajes, y la respuesta generada con su latencia.
Sin esta traza es casi imposible diagnosticar por qué una respuesta salió mal. En un contexto
educativo, estos registros también sirven para demostrar que el sistema efectivamente
recuperó información real de los documentos.

## Chequeo de temperatura y determinismo

La temperatura controla el grado de aleatoriedad en el muestreo del modelo. Para tareas de
recuperación de información conviene una temperatura baja, cercana a cero, que favorezca
respuestas fieles y reproducibles. Una temperatura alta puede producir respuestas más
creativas pero menos apegadas al contexto, lo cual es indeseable en un sistema que promete
responder solo con base en los documentos provistos.
