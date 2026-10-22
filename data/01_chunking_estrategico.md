# Chunking Estratégico en Pipelines RAG

## Qué es el chunking y por qué define la calidad del sistema

El chunking es el proceso de fragmentar un documento largo en unidades más pequeñas,
llamadas chunks, que puedan ser embebidas de forma independiente y recuperadas luego
durante la consulta. En un pipeline de Retrieval-Augmented Generation (RAG), el chunking
no es un detalle de implementación: es la decisión que más impacto tiene sobre la calidad
final de las respuestas. Un modelo de embeddings comprime un fragmento de texto en un
vector de dimensión fija (por ejemplo, 1536 dimensiones en el modelo text-embedding-3-small).
Ese vector captura una idea agregada del fragmento completo, no cada palabra. Si el
fragmento contiene cinco ideas distintas, el vector será un promedio difuso de todas ellas
y la búsqueda por similitud perderá precisión.

## Fixed-size chunking

La estrategia más simple es el chunking de tamaño fijo. Se define una cantidad de
caracteres o de tokens por fragmento y se corta el texto cada esa cantidad. Es rápida y
predecible, pero tiene un defecto grave: corta oraciones y párrafos por la mitad. Un corte
en medio de una idea rompe la coherencia semántica del fragmento y separa el sujeto de
su predicado, degradando la representación vectorial.

## Chunking recursivo

La estrategia recomendada en la mayoría de los casos es el chunking recursivo, implementado
en LangChain por la clase RecursiveCharacterTextSplitter. Esta clase recibe una lista
ordenada de separadores, por ejemplo ["\n\n", "\n", " ", ""], y los aplica en cascada.
Primero intenta dividir por párrafos con el separador de doble salto de línea. Si un
párrafo todavía supera el tamaño objetivo, vuelve a dividirlo por líneas simples. Si aún
es demasiado grande, divide por espacios, y como último recurso corta carácter por carácter.
El resultado es que respeta la estructura natural del texto siempre que sea posible y solo
recurre al corte brusco cuando no hay otra opción.

## El parámetro chunk_size

chunk_size define el tamaño máximo de cada fragmento, medido en caracteres por defecto o en
tokens si se configura un length_function basado en tiktoken. Un valor habitual para textos
técnicos es de 500 a 1000 tokens. Fragmentos muy chicos (menos de 200 tokens) pueden no
contener suficiente contexto para responder una pregunta y obligan a recuperar muchos
fragmentos. Fragmentos muy grandes (más de 2000 tokens) diluyen el embedding y consumen
presupuesto de contexto del modelo de lenguaje.

## El parámetro chunk_overlap

chunk_overlap define cuántos caracteres o tokens comparten dos fragmentos consecutivos. El
solapamiento es la defensa contra la pérdida de información en las fronteras. Si una idea
cruza el límite entre el chunk A y el chunk B, con overlap esa idea aparece completa en al
menos uno de los dos. Un valor típico es del 10 al 20 por ciento del chunk_size; por ejemplo,
chunk_size 500 con chunk_overlap 50. Un overlap demasiado alto infla el tamaño del índice
porque almacena texto duplicado, sin mejorar la recuperación de forma proporcional.

## Cuantificación con tiktoken

Para medir el chunk_size en tokens reales del modelo y no en caracteres, se pasa una
length_function que use el tokenizador tiktoken. Esto es importante porque distintos
modelos tokenizan distinto y porque el costo de ingesta y de consulta se factura por token.
Medir en tokens evita sorpresas de límite de contexto y de facturación.

## Errores frecuentes de chunking

El primer error es usar solo el splitter de caracteres sin pensar en la semántica del
documento. El segundo es fijar chunk_size y chunk_overlap arbitrariamente sin medir cuántos
tokens contiene realmente cada archivo fuente. El tercero es no preservar metadatos: cada
chunk debería conservar la referencia al archivo de origen y, si es posible, a la sección,
para poder citar la fuente en la respuesta final. Un pipeline serio etiqueta cada chunk con
source, chunk_index y cualquier metadata útil para la trazabilidad.

## La conexión con la recuperación

El chunking y la recuperación están acoplados. Si los chunks son de 500 tokens y el
retriever devuelve un top_k de 4, el modelo recibe alrededor de 2000 tokens de contexto,
que es una carga razonable y bien enfocada. Si los chunks fueran de 3000 tokens, el mismo
top_k de 4 entregaría 12000 tokens y probablemente activaría el fenómeno conocido como
Lost in the Middle, donde el modelo presta menos atención a la información ubicada en el
centro de un contexto muy largo.
