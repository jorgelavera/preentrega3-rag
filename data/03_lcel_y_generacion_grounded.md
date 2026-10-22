# Cadenas LCEL y Generación Grounded

## Qué es LCEL

LCEL, LangChain Expression Language, es un lenguaje de composición declarativo para construir
cadenas en LangChain. Su operador central es el pipe, el símbolo |, que conecta la salida de
un componente con la entrada del siguiente. La expresión prompt | model | parser se lee como
una tubería de datos. LCEL funciona porque cada componente implementa la interfaz Runnable,
que garantiza métodos uniformes como invoke, ainvoke, batch y stream. Esa uniformidad es lo
que hace que cualquier Runnable pueda conectarse con cualquier otro.

## Prompts en LCEL

El primer eslabón suele ser una plantilla de prompt, como ChatPromptTemplate, construida a
partir de mensajes con roles. El rol system define el comportamiento y las reglas del modelo;
el rol human contiene la pregunta del usuario y, en RAG, el contexto recuperado. Los
placeholders como {context} y {question} se rellenan en tiempo de ejecución cuando la cadena
recibe un diccionario de entrada.

## El modelo como Runnable

El modelo de lenguaje, por ejemplo ChatOpenAI, expone métodos síncronos y asíncronos a través
de la interfaz Runnable. El método ainvoke es la versión asíncrona de invoke y es lo que
permite construir pipelines no bloqueantes. La versión asíncrona es especialmente valiosa
porque una llamada a un LLM por red es una operación de entrada y salida de alta latencia, y
bloquear el hilo principal mientras se espera desperdicia recursos.

## Asincronía en pipelines RAG

El módulo asyncio de Python y la función asyncio.run permiten ejecutar corrutinas. Una
corrutina se declara con async def y se espera con await. En una cadena RAG, tanto la
búsqueda en la base vectorial como la llamada al modelo pueden ser asíncronas. Esto abre la
puerta a procesar varias consultas en paralelo con asyncio.gather, reduciendo el tiempo total
cuando hay muchos usuarios. Para que la asincronía rinda de verdad, todos los eslabones deben
soportar await; llamar a una función síncrona dentro de una corrutina bloquea el event loop.

## Generación grounded

La generación grounded es la práctica de obligar al modelo a responder únicamente a partir
del contexto recuperado, sin usar su conocimiento paramétrico previo. Es la diferencia entre
un chatbot que inventa y un sistema de recuperación de información confiable. El prompt de
sistema es el mecanismo para lograr groundedness: debe instruir explícitamente al modelo a
basar su respuesta solo en el contexto y a admitir ignorancia cuando la respuesta no esté
allí.

## El prompt como filtro de veracidad

Un buen prompt de sistema para RAG tiene varias reglas. Primero, define el rol, por ejemplo
asistente técnico. Segundo, ordena usar exclusivamente el CONTEXTO entregado. Tercero,
indica que si la respuesta no está en el contexto, el modelo debe decir que no tiene acceso
a esa información en lugar de inventar. Cuarto, y esto es clave para la verificabilidad,
exige que el modelo cite el archivo de origen del que tomó cada afirmación. Un prompt que
solo dice responde la pregunta, sin imponer estas restricciones, invita a la alucinación.

## La alucinación y cómo mitigarla

La alucinación es la generación de información falsa presentada con seguridad. En RAG se
mitiga de tres formas combinadas. La primera es la recuperación misma: entregar contexto
relevante reduce la necesidad del modelo de inventar. La segunda es el prompt restrictivo:
ordenar decir no lo sé cuando no hay contexto suficiente. La tercera es la baja temperatura:
valores cercanos a cero reducen la creatividad del muestreo y favorecen respuestas fieles al
texto. Ninguna de las tres es infalible por separado, pero juntas reducen mucho el riesgo.

## Lost in the Middle

Lost in the Middle es un fenómeno documentado en modelos con ventanas de contexto muy
grandes. Cuando se entrega un contexto enorme, el modelo presta más atención al principio y
al final y menos a la información ubicada en el medio. La consecuencia práctica es
contraintuitiva: pasar más fragmentos puede empeorar la respuesta. Por eso conviene mantener
un top_k acotado de 3 a 5 fragmentos en lugar de inyectar decenas de documentos.

## Salida estructurada con Pydantic

Los LLM producen texto libre, pero las aplicaciones necesitan datos estructurados. Pydantic
permite definir un modelo con campos tipados, por ejemplo un campo answer de tipo texto y un
campo sources de tipo lista. El PydanticOutputParser de LangChain genera las instrucciones de
formato que se inyectan en el prompt y luego valida y parsea la respuesta del modelo contra
el modelo Pydantic. Si el modelo devuelve algo que no cumple el esquema, el parser lanza un
error, lo que permite detectar respuestas mal formadas en lugar de romper silenciosamente.

## Por qué estructurar la salida

Estructurar la salida con un esquema trae tres beneficios. Primero, el contrato es explícito:
quien consume la función sabe exactamente qué campos recibirá. Segundo, se puede separar el
texto de la respuesta de las referencias a las fuentes, lo que facilita mostrar citas en una
interfaz. Tercero, la validación automática actúa como una red de seguridad contra respuestas
mal formadas. En el caso de este tipo de sistema, el modelo Pydantic incluye el texto de la
respuesta y la lista de archivos de origen citados.
