# Guion oral: Delivery II (10 min)

OneRef · IA025 · Unicamp 2026 · Peter Roque Vilchez & Charles Maxime Doudy

Presentación: `OneRef – Preliminary Results (Delivery II).pdf` (11 diapositivas, en inglés). El texto de las diapositivas citado abajo queda en inglés; el guion oral está en español. Versiones: inglés `speaker_script_delivery_II.en.md`, francés `speaker_script_delivery_II.fr.md`.

## Tiempos y presentadores

Unas 1 420 palabras habladas, aproximadamente 10 min a un ritmo normal en español (~145 palabras/min). Si se pasan del tiempo en el ensayo, acorten primero las diapositivas 4 y 7.

| # | Diapositiva | Presentador | Tiempo |
|---|---|---|---|
| 1 | Portada | Charles | 25 s |
| 2 | Contexto | Charles | 60 s |
| 3 | Lo que construimos | Charles | 60 s |
| 4 | Primeros números reales | Peter | 75 s |
| 5 | Lo que funcionó y lo que no | Peter | 50 s |
| 6 | Lo que salió mal | Peter | 60 s |
| 7 | Primera ejecución de punta a punta | Charles | 70 s |
| 8 | Ejemplos de la ejecución | Charles | 55 s |
| 9 | Planeado vs. hecho | Peter | 40 s |
| 10 | Próximos pasos | Peter | 45 s |
| 11 | Gracias | Charles | 10 s |

Charles (unos 4,7 min) presenta el contexto, el código de los lectores, el verificador, el agregador y el orquestador, y la ejecución de punta a punta. Peter (unos 4,5 min) presenta el experto visual DINOv3, los datos y los splits, los problemas encontrados, el balance frente al plan y los próximos pasos. Cada uno responde las preguntas del jurado sobre su parte.

## Para saber de memoria

1. **DINOv3 no es mejor en AUROC.** Sobre todas las imágenes de dev, v3 obtiene **0,59** y v2 **0,61**. Nos quedamos con v3 porque el trabajo del experto es *localizar*: su pico cae sobre el defecto el **77 %** de las veces en MVTec AD, frente al **51 %** de v2. GoodsAD (296 de las 835 imágenes) baja el promedio global.
2. **La recalibración 0,49 → 0,61 es in-sample.** Se ajustó y se midió en dev, así que es optimista. Decirlo antes de que lo diga el jurado.
3. **71,2 % en 18 imágenes no es un resultado.** El IC del 95 % es **[54 %, 86 %]**. La ejecución usó Qwen3-VL-4B cuantizado como ambos lectores y un verificador solo con reglas. Nunca compararlo con 73,9 %.
4. **La discriminación está en 52,5 % balanceada, cerca del azar (50 %).** Es la principal debilidad por ahora, coherente con el AUROC bajo del experto fuera de MVTec AD.
5. **20/20 vs 21/21.** La lección de la diapositiva 6 habla de 20 tests aprobados con datos falsos; la diapositiva 3 dice 21. El 21.º es el test del cargador con los 9 nombres reales, agregado después del fallo.
6. **Los números cuadran.** 835 + 7 531 = 8 366 imágenes. 184 llamadas = 2 lectores × 88 preguntas + 4 reaperturas × 2 lectores. "~3,7 s por pregunta" es por llamada de lector: (323 + 322) / 176 ≈ 3,7 s. Dev tiene 3 937 preguntas.
7. **73,9 % para Qwen3-VL-8B es un número reportado** (artículo de AD-Copilot; SiGMA mide 74,5 %). B1 todavía no se volvió a medir en nuestro split: es la primera tarea de esta semana.

## Cambios de presentador

- 3 → 4, Charles: "Peter va a mostrar ahora lo que hace el experto con los datos reales."
- 6 → 7, Peter: "Charles va a presentar esa primera ejecución de punta a punta."
- 8 → 9, Charles: "Peter, ¿dónde estamos frente al plan?"
- 10 → 11, Peter: "¿Charles?"

---

## 1 · Portada: OneRef

### Contenido visual
- Encabezado: IA025 · Deep Learning · Unicamp 2026
- Título: **OneRef**
- Subtítulo: preliminary results: the pipeline runs end to end, and the visual expert is measured on real MMAD data
- Equipo: Peter Roque Vilchez · Charles Maxime Doudy · Delivery II · November 12, 2026
- Imágenes: una rejilla metálica de MVTec AD ("Part") y el mapa de anomalía de nuestro experto, con el pico rojo sobre el defecto

### Guion oral — Charles (~25 s, ~65 palabras)
Buenos días. Soy Charles Maxime Doudy y, junto con Peter Roque Vilchez, presentamos la Delivery II de OneRef, nuestro proyecto de IA025. Hace dos semanas les mostramos el plan. Hoy les mostramos dónde estamos: el pipeline funciona de punta a punta, y el experto visual está medido con datos reales de MMAD. A la derecha, una imagen real de dev y el mapa de nuestro experto: la mancha roja es el defecto.

### Añadir / mejorar
- Agregar la pregunta de investigación en una línea: "How much of the multi-agent gain survives with one reference and no tuning?"
- Rotular la imagen de arriba "Part (MVTec AD grid, dev)" para que el jurado sepa que son datos reales y no una ilustración.

### Pregunta del jurado
**"¿Qué cambió desde la Delivery I?"**
Tres cosas: todo el workflow ya existe como código, el experto visual corrió sobre las 835 imágenes de dev con puntajes medidos por fuente, y los cinco agentes corrieron juntos por primera vez sobre imágenes reales. Lo que no cambió es el protocolo: una sola referencia, sin entrenamiento, y el test se corre una sola vez.

---

## 2 · Contexto: one good photo, one part, a question

### Contenido visual
- Referencia (botella sana) y pieza (botella con el borde roto) → **1 · Visual expert** → **2 · Two readers** → **3 · Verifier** → **4 · Vote**
- Benchmark: MMAD, **8 366 imágenes**, **39 670** preguntas de opción múltiple, **7 subtareas**, strict 1-shot
- Baseline a superar: Qwen3-VL-8B solo, **73,9 %**, y un voto por mayoría con el mismo costo
- Reglas: sin entrenamiento, modelos **≤ 8B**, ajustes solo en dev, test corrido una sola vez
- Pie: hipótesis = superar ambas baselines con significancia (McNemar); un resultado negativo bien medido también es válido

### Guion oral — Charles (~60 s, ~140 palabras)
Un breve recordatorio del problema. El sistema recibe una foto de una pieza sana, la foto de la pieza a inspeccionar y una pregunta de opción múltiple. Responde en cuatro etapas: un experto visual compara las dos fotos, dos lectores, que son modelos de lenguaje multimodales, responden la pregunta, un verificador los controla y un voto decide. Evaluamos en MMAD: 8 366 imágenes, 39 670 preguntas, siete subtareas, y nos mantenemos estrictamente en one-shot, con una sola referencia por pregunta. La baseline a superar es Qwen3-VL-8B solo, reportado en 73,9 %, y un voto por mayoría con el mismo presupuesto de cómputo. Las reglas no cambiaron: sin entrenamiento, modelos de como máximo 8 mil millones de parámetros, todas las decisiones tomadas en dev, y el test corrido una sola vez. Un resultado negativo bien medido también es válido.

### Añadir / mejorar
- Escribir "73.9 % (reported)" y agregar "re-measured on our split", para que nadie lo tome como una medición nuestra.
- Reemplazar "strict 1-shot" por "1 reference image per question": "1-shot" puede leerse como un ejemplo de entrenamiento.

### Pregunta del jurado
**"¿Por qué valdría un resultado negativo?"**
Porque la pregunta es cuánto del 85,2 % de SiGMA sobrevive con un protocolo estricto. SiGMA usa hasta 256 imágenes normales, texto de dominio de MMAD y un equipo elegido con las respuestas de MMAD. Si la ganancia desaparece con una sola referencia y sin ajuste, es una medición útil, siempre que venga con un intervalo de confianza y un análisis de errores.

---

## 3 · Lo que construimos: the full workflow exists as code

### Contenido visual
- Cinco tarjetas: **Visual expert** (DINOv3, distancia entre parches, *ran on all 835 dev*) · **Readers A · B** (Qwen3-VL · InternVL3 vía vLLM) · **Verifier** (reglas + crítico LLM, reapertura ≤ 1) · **Aggregator** (voto ponderado por subtarea) · **Orchestrator** (un modelo por pasada, caché, reanudación); las cuatro últimas: *ran on real data*
- **21 / 21** tests sin GPU aprobados (esquemas, crítico, voto, métricas, reanudación)
- **835 / 7 531** imágenes dev / test, estratificadas, semilla fija y commiteada
- **2,1 GB** para descargar dev en lugar de 28 GB
- Pie: JSON validado con Pydantic entre agentes, un reintento y luego error; código en github.com/RVilchezzz/OneRef

### Guion oral — Charles (~60 s, ~140 palabras)
Primer mensaje: todo lo que estaba en el plan ya existe como código. Cinco componentes. El experto visual DINOv3. Los lectores A y B, Qwen3-VL e InternVL3, servidos con vLLM. El verificador, que combina reglas fijas con un crítico LLM y puede reabrir un conflicto una sola vez como máximo. El agregador, un voto ponderado por subtarea. Y el orquestador, que carga un solo modelo por pasada, guarda cada salida en disco y puede reanudar después de una caída. Los agentes solo intercambian JSON validado con Pydantic: una respuesta inválida se reintenta una vez y luego cuenta como error. Tres números: 21 tests pasan sin GPU; dev y test están fijados en 835 y 7 531 imágenes, estratificadas, con una semilla commiteada; y descargamos 2,1 gigabytes para dev en lugar de 28, leyendo solo los archivos necesarios dentro de los zips remotos. Peter va a mostrar ahora lo que hace el experto con los datos reales.

### Añadir / mejorar
- "Ran on real data" en los lectores y el verificador es cierto, pero necesita una aclaración: escribir "ran on real data (4B, 18 images)". Si no, el jurado pensará que los lectores 8B ya corrieron.
- Alinear el número de tests con la diapositiva 6 (21 aquí, 20 allá): ver el punto 5 de "Para saber de memoria".

### Pregunta del jurado
**"¿Por qué un modelo por pasada en lugar de un pipeline en tiempo real?"**
Porque la GPU solo aguanta un modelo de 8B: 6 GB en local, y una L4 o una T4 en Colab. Cada agente recorre todo el split, escribe sus salidas en disco y libera la GPU antes del siguiente. Además, todas las baselines reutilizan las mismas salidas de los lectores en caché, así que B1 a B4 casi no cuestan nada extra.

---

## 4 · Primeros números reales: the expert works on aligned photos only

### Contenido visual
| Source (dev) | Images | AUROC v2 | AUROC v3 | In mask v2 | In mask v3 |
|---|---|---|---|---|---|
| MVTec AD | 166 | 0,88 | 0,84 | 51 % | **77 %** |
| VisA | 215 | 0,77 | 0,78 | 42 % | 41 % |
| MVTec LOCO | 156 | 0,64 | 0,62 | 37 % | 43 % |
| GoodsAD | 296 | 0,55 | 0,51 | 12 % | 10 % |
| All | 835 | 0,61 | 0,59 | 33 % | 40 % |
- Tarjetas: "DINOv3 localizes better: 77 % vs 51 % on MVTec AD" y "GoodsAD ≈ chance: the one reference is often another product"
- Pie: DINOv2-base y DINOv3 ViT-B/16 congelados, consulta vs `random_templates[0]`, solo dev, una sola ejecución, **0,35 s** por imagen en la RTX 3060 Laptop; máscaras usadas solo para puntuar

### Guion oral — Peter (~75 s, ~180 palabras)
Gracias, Charles. El experto visual compara la pieza con su única referencia usando features DINO congeladas. Para cada parche de la pieza, la distancia al parche más cercano de la referencia da un mapa de anomalía. Probamos dos backbones, DINOv2 y DINOv3, en las 835 imágenes de dev, con dos medidas. El AUROC es la probabilidad de que una pieza defectuosa tenga un puntaje más alto que una normal: 0,5 es el azar. "In mask" es la frecuencia con la que el punto más caliente del mapa cae sobre el defecto real. En MVTec AD, donde las fotos están alineadas, el experto funciona: AUROC por encima de 0,84, y DINOv3 pone su pico sobre el defecto el 77 % de las veces, frente al 51 % de DINOv2. Por eso nos quedamos con DINOv3: localizar es el trabajo del experto, aunque su AUROC global sea un poco más bajo, 0,59 frente a 0,61. Fuera de MVTec AD, empeora: VisA alrededor de 0,78, LOCO alrededor de 0,63, y GoodsAD está al nivel del azar, porque la única referencia suele ser otro producto. Costo: 0,35 segundos por imagen en nuestra laptop.

### Añadir / mejorar
- El encabezado "Images" se corta en "Image / s": ensanchar la columna.
- Poner en negrita el mejor valor de cada par para que el ojo lo encuentre enseguida.
- Agregar una nota bajo "All": la fila global está ponderada por imágenes, y GoodsAD es el 35 % de dev (296 / 835).

### Pregunta del jurado
**"¿Por qué quedarse con DINOv3 si su AUROC es más bajo?"**
La diferencia de AUROC es pequeña (0,59 frente a 0,61, una sola ejecución) y viene sobre todo de GoodsAD, donde ambos están al nivel del azar. En localización, que es lo que los lectores reciben como pista, DINOv3 es claramente mejor en MVTec AD (77 % frente a 51 %) y en LOCO (43 % frente a 37 %). La detección en sí queda en manos de los lectores y del voto, no solo del puntaje del experto.

---

## 5 · Lo que funcionó y lo que no: reference · part · our DINOv3 map

### Contenido visual
- A la izquierda, **✓ Worked: defect found**: rejilla de MVTec AD; chicle de VisA (referencia, pieza, mapa)
- A la derecha, **× Failed: reference is another product**: caja de alimentos de GoodsAD (referencia verde, pieza rosa); jugo de MVTec LOCO (etiqueta naranja frente a etiqueta de plátano)
- Pie: imágenes reales de dev; licencias de MMAD (CC BY-NC-SA 4.0, CC BY 4.0, uso de investigación)

### Guion oral — Peter (~50 s, ~125 palabras)
Así se ve en la práctica. Cada fila muestra la referencia, la pieza y nuestro mapa DINOv3. A la izquierda, dos aciertos: una rejilla metálica de MVTec AD y un chicle de VisA. La mancha roja cae justo sobre el defecto. A la derecha, dos fallos con la misma causa. En GoodsAD, la referencia es una caja verde y la pieza una caja rosa: el mapa ilumina todo el producto, porque todo es distinto. En MVTec LOCO, el jugo de referencia tiene una etiqueta naranja y la pieza una etiqueta de plátano: el experto sí ve una diferencia, pero no sabe distinguir una variante del producto de un defecto. La lección: con una sola referencia, el experto vale lo que vale la coincidencia entre las dos fotos.

### Añadir / mejorar
- Poner una pequeña etiqueta encima de cada columna: "Reference · Part · Map", repetida en ambos lados.
- Debajo de cada caso fallido, agregar el puntaje del experto junto al umbral de decisión, para que el jurado vea la falsa alarma en números.

### Pregunta del jurado
**"¿Por qué no elegir una mejor referencia, por ejemplo la imagen normal más parecida?"**
Eso sería usar `similar_templates` o una galería de imágenes normales, que es justo lo que hace SiGMA y lo que nuestro protocolo estricto prohíbe. Solo usamos `random_templates[0]`. El objetivo es medir lo que sobrevive con una referencia que el sistema no eligió; una mejor referencia puede ser una variante aparte, claramente etiquetada.

---

## 6 · Lo que salió mal: four problems found, and what we do about them

### Contenido visual
| Problem | Evidence | Fix / plan |
|---|---|---|
| Loader never tried on real data | crashed on "Anomaly Detection", the name `mmad.json` uses | fixed + test with the 9 real names |
| Expert threshold off-scale for DINOv3 | distances **0.11–0.31** vs range 0.2–0.8: flag never fires (bal. acc. **0.49**) | recalibrated on dev: **0.49 → 0.61** (in-sample) |
| Single reference ≠ same product | GoodsAD AUROC **0.51**, peak on defect **10 %** | trust the expert less there: weights fixed on dev |
| 8B readers not run yet | Colab not executed; local 4B run: **18 images** only | 50-question cost test, then full dev |
- Lección: unit tests with fake data passed 20/20 while the real file broke the first line. Run on real data early.

### Guion oral — Peter (~60 s, ~145 palabras)
Aparecieron cuatro problemas, y esto es lo que hicimos con cada uno. Uno: nuestro cargador de datos nunca había visto el archivo real. Falló con el nombre de categoría "Anomaly Detection", que es el que realmente usa mmad.json. Ya está corregido, con un test sobre los nueve nombres reales. Dos: el umbral del experto estaba pensado para distancias entre 0,2 y 0,8, pero las de DINOv3 caen entre 0,11 y 0,31, así que la alerta nunca se activaba: precisión balanceada de 0,49. Lo recalibramos en dev y llegamos a 0,61, pero está medido sobre las mismas imágenes, así que es optimista. Tres: la única referencia a menudo no es el mismo producto, como en GoodsAD, así que el voto confiará menos en el experto ahí. Cuatro: los lectores de 8B todavía no corrieron. La lección: nuestros tests unitarios pasaban con datos falsos mientras el archivo real fallaba en la primera línea. Charles va a presentar esa primera ejecución de punta a punta.

### Añadir / mejorar
- Escribir "20/20 at the time" en la lección, o actualizarla a 21/21, para que coincida con la diapositiva 3.
- Poner "(in-sample)" justo al lado del 0,61, en naranja: es la aclaración honesta que el jurado va a buscar.

### Pregunta del jurado
**"¿Recalibrar el umbral en dev no es sobreajuste?"**
Es un ajuste en dev, y para eso sirve dev: el protocolo prohíbe ajustar en el test, no en dev. El 0,61 es in-sample, así que lo presentamos como una estimación alta. La verdadera prueba es la ejecución del test del 19 de noviembre, con el umbral congelado, y ahí también reportaremos la precisión balanceada del experto.

---

## 7 · Primera ejecución de punta a punta: all five agents ran on real MMAD images

### Contenido visual
- Flujo: Expert **18 s** → Reader A **323 s** → Reader B **322 s** → Verifier: **4 reopened** → Vote
- Precisión sobre **88 preguntas**: discriminación (balanceada) **52,5 %** · clasificación **62,5 %** · descripción **62,5 %** · localización **50,0 %** · análisis **87,5 %** · clasificación de objeto **100 %** · análisis de objeto **83,3 %** · **media de las 7: 71,2 %**
- Tarjetas: **0 errors in 184 calls** (todo JSON válido, ~3,7 s por pregunta) · **Readers agree 87/88** (el mismo modelo dos veces) · **95 % CI [54 %, 86 %]** (18 imágenes: un pipeline que funciona, no un resultado)
- Pie: prueba de humo en la laptop, RTX 3060 6 GB, 18 imágenes de dev (2 a 4 por fuente), ambos lectores = Qwen3-VL-4B cuantizado (Ollama), verificador solo con reglas; los lectores de 8B corren en Colab

### Guion oral — Charles (~70 s, ~165 palabras)
Gracias, Peter. Esta es nuestra primera ejecución de punta a punta: los cinco agentes, sobre imágenes reales de MMAD, en la GPU de 6 gigabytes de la laptop. Dieciocho imágenes de dev, de dos a cuatro por fuente, 88 preguntas. Para que entrara en memoria, ambos lectores fueron el mismo modelo pequeño, Qwen3-VL-4B cuantizado, y el verificador usó solo reglas. Arriba: 18 segundos para el experto, unos 320 segundos por lector, y el verificador reabrió cuatro conflictos. A la izquierda, la precisión por subtarea, con una media de 71,2 %. La discriminación está en 52,5 % balanceada, cerca del azar, lo que coincide con el AUROC bajo del experto. A la derecha, lo que realmente aprendimos. Cero errores en 184 llamadas: cada respuesta fue un JSON válido. Los dos lectores coincidieron en 87 de 88 preguntas, porque era el mismo modelo dos veces, y justamente por eso el sistema real usa dos familias distintas. Y el intervalo de confianza va de 54 a 86 %: esto es un pipeline que funciona, no un resultado.

### Añadir / mejorar
- Agregar el número de preguntas por subtarea junto a cada puntaje: con 88 preguntas, algunas celdas tienen menos de diez, y 100 % en clasificación de objeto dice poco.
- Escribir "~3.7 s per reader call" en lugar de "per question" (323 + 322 s para 176 llamadas).
- Poner una línea punteada en 50 % junto al puntaje de discriminación.

### Pregunta del jurado
**"Con 52,5 % en discriminación, ¿el experto no empeora las cosas?"**
No se puede saber con 18 imágenes, y eso es exactamente lo que mide la ablación "sin experto". Lo que sabemos es que el experto es confiable en fotos alineadas y está al nivel del azar en GoodsAD, así que su voto se ponderará por subtarea según su confiabilidad en dev. Si aun así empeora los resultados, la ablación lo mostrará y lo reportaremos.

---

## 8 · Ejemplos de la ejecución: four real cases, four behaviours

### Contenido visual
- Cada tarjeta: referencia · pieza · el zoom que el experto envió a los lectores (en naranja)
- **✓ Cable** (MVTec AD): 5/5 correctas, "Missing component, in the central part"
- **↻ Tile** (MVTec AD): el verificador reabrió; lectores "bottom-left", experto "bottom-center"; sigue mal (center-left)
- **× Can** (GoodsAD): falsa alarma, pieza sana declarada defectuosa; la referencia es otra marca
- **× Pushpins** (MVTec LOCO): siguió la pista, "the expert hint indicates the top right": lugar equivocado
- Takeaway: la pista del experto es poderosa en ambos sentidos, así que su peso debe depender de su confiabilidad en dev

### Guion oral — Charles (~55 s, ~140 palabras)
Cuatro casos reales de esa ejecución, y cuatro comportamientos distintos. Cada tarjeta muestra la referencia, la pieza y, en naranja, el zoom que el experto envió a los lectores. Arriba a la izquierda, un cable de MVTec AD: cinco preguntas de cinco correctas, con la respuesta "componente faltante, en la parte central". Arriba a la derecha, una baldosa: los lectores dijeron abajo a la izquierda, el experto abajo al centro. El verificador detectó el conflicto y lo reabrió, pero la respuesta final siguió siendo incorrecta: el defecto está en el centro a la izquierda. Abajo a la izquierda, una lata de GoodsAD: falsa alarma, porque la referencia es otra marca. Abajo a la derecha, chinches de LOCO: el lector escribió "la pista del experto indica arriba a la derecha" y la siguió hasta el lugar equivocado. La conclusión: la pista del experto es poderosa en ambos sentidos, así que su peso debe depender de su confiabilidad en dev. ¿Peter?

### Añadir / mejorar
- En la tarjeta de la baldosa, agregar la posición real ("truth: center-left") en otro color, para que la historia de la reapertura se lea de un vistazo.
- En la tarjeta de las chinches, recordar que los defectos de LOCO son lógicos (cantidad incorrecta, pieza faltante): una distancia entre parches no puede verlos.

### Pregunta del jurado
**"El verificador reabrió la baldosa y la respuesta siguió mal. ¿Para qué sirve?"**
Su trabajo es detectar un desacuerdo y forzar una segunda mirada, no saber la respuesta. En esta ejecución usaba solo reglas y los dos lectores eran el mismo modelo 4B, así que la segunda mirada no aportaba mucho nuevo. Con dos familias de modelos y el crítico LLM, la ablación "sin verificador" medirá lo que aporta; cuatro reaperturas en 88 preguntas también muestran que no entra en bucle.

---

## 9 · Planeado vs. hecho: what the plan promised for Nov 12

### Contenido visual
| Planned (Delivery I) | Status | Detail |
|---|---|---|
| MMAD download | Done | dev subset, **2.1 GB**, **2,084 files** |
| Dev and test splits fixed | Done | **835 / 7,531** images, committed |
| DINOv3 expert with anomaly maps | Done | all 835 dev images, per-source scores |
| Evaluation harness | Partial | metrics, McNemar, bootstrap coded; check vs official `summary.py` pending |
| B1 and B2 on dev | Pending | readers coded and tested, not run |
| Colab throughput measured | Pending | first task this week |
- **3 / 6** ítems hechos · **+ extra**: todo el workflow multiagente codificado y ya corrido de punta a punta sobre 18 imágenes de dev (planeado para el 12–19 nov.)

### Guion oral — Peter (~40 s, ~100 palabras)
Frente a lo que prometimos para hoy: tres de seis puntos están hechos. La descarga de dev de MMAD, 2,1 gigabytes y 2 084 archivos. Los splits, fijados y commiteados. Y el experto DINOv3 sobre todas las imágenes de dev, con puntajes por fuente. El arnés de evaluación está parcial: las métricas, McNemar y el bootstrap están codificados, pero aún no verificados contra el script oficial de MMAD. B1 y B2 en dev, y el rendimiento en Colab, siguen pendientes. En cambio, todo el workflow multiagente, planeado para la próxima semana, ya está codificado y corrió de punta a punta.

### Añadir / mejorar
- Escribir "3 done · 1 partial · 2 pending" en lugar de "3 / 6": es más honesto y se lee más rápido.
- Decir por qué B1 y B2 se atrasaron (Colab aún no ejecutado) en la columna Detail.

### Pregunta del jurado
**"Están atrasados con las baselines. ¿Es realista congelar el 19 de noviembre?"**
Los lectores están codificados y probados; lo que falta es tiempo de GPU, no código. La prueba de costo con 50 preguntas fija el tamaño del test: de 1 000 a 2 000 imágenes según las unidades medidas. Todas las baselines reutilizan las mismas salidas de los lectores en caché, así que B3 y B4 salen casi gratis una vez que B1 y B2 hayan corrido.

---

## 10 · Próximos pasos: same goal, adjusted path to Nov 26

### Contenido visual
- **Nov 12–15 · Readers on dev**: prueba de costo con 50 preguntas en Colab L4, luego B1 y B2 sobre las **3 937** preguntas de dev
- **Nov 15–19 · Calibrate on dev**: umbral del experto, pesos del voto por lector y subtarea, prompts; verificación con el `summary.py` oficial
- **Nov 19 · Freeze · Test, run once**: S frente a B1–B4, ablaciones; **~1 000–2 000** imágenes de test según el costo medido
- **Nov 26 · Final**: McNemar + bootstrap, análisis de errores por fuente, informe
- Ajuste: el voto del experto dependerá de su confiabilidad en dev para cada subtarea; resultados siempre reportados por fuente (GoodsAD y LOCO aparte)

### Guion oral — Peter (~45 s, ~120 palabras)
El objetivo no cambia, pero el camino se ajusta. Hasta el 15: la prueba de costo con 50 preguntas en una L4 de Colab, y luego B1 y B2 sobre las 3 937 preguntas de dev. Del 15 al 19: la calibración en dev, es decir, el umbral del experto, los pesos del voto por lector y por subtarea, los prompts, y la verificación con el script oficial. El 19 congelamos todo y corremos el test una sola vez: el sistema completo frente a B1 a B4, más las ablaciones, sobre 1 000 a 2 000 imágenes según el costo medido. El 26: McNemar, bootstrap, análisis de errores por fuente y el informe. Un ajuste: el voto del experto dependerá de su confiabilidad en dev. ¿Charles?

### Añadir / mejorar
- Agregar el presupuesto de Colab (100 unidades, nunca A100) y el plan B (+100 unidades) bajo la primera caja.
- Marcar el congelamiento como una línea dura en la línea de tiempo, por ejemplo "no change after this point".

### Pregunta del jurado
**"¿Y si la prueba con 50 preguntas muestra que Colab es demasiado caro?"**
Entonces el test se reduce hacia 1 000 imágenes, que siguen dando suficientes preguntas pareadas para McNemar, y pasamos a la T4 en 4 bits si la L4 consume demasiadas unidades. El costo se mide antes de fijar el tamaño del test, justamente para que nunca nos obligue a cambiar el protocolo.

---

## 11 · ¡Gracias!

### Contenido visual
- Encabezado: OneRef · Delivery II
- **Thank you! Questions?**
- Código y resultados: github.com/RVilchezzz/OneRef
- Imagen: mapa DINOv3 sobre el chicle de VisA

### Guion oral — Charles (~10 s, ~30 palabras)
Muchas gracias por su atención. El código y los resultados son públicos en github.com/RVilchezzz/OneRef. Con gusto respondemos sus preguntas.

### Añadir / mejorar
- Nada que cambiar. Tener lista la diapositiva 4 para volver a ella: la mayoría de las preguntas serán sobre eso.

### Pregunta del jurado
**"¿Cuál es el número que debemos recordar hoy?"**
77 %: en fotos alineadas, el pico de nuestro experto cae sobre el defecto real tres de cada cuatro veces, a partir de una sola referencia. Todo lo demás, incluso si el equipo completo supera a Qwen3-VL-8B, se medirá en el test después del 19 de noviembre.

---

## Preguntas de respaldo

**"¿Por qué no hacer fine-tuning del experto o de los lectores en MMAD?"**
El protocolo es sin entrenamiento por diseño: MMAD es nuestro banco de pruebas, y entrenar en él dejaría sin sentido la comparación con el leaderboard. Todo lo entrenado en MMAD pertenece a otra categoría.

**"¿Un AUROC de 0,59 alcanza para ser útil?"**
No como detector por sí solo, y no lo usamos así. En MVTec AD llega a 0,84–0,88; en el global lo baja GoodsAD. Su valor principal es la localización, que se pasa a los lectores como un zoom y una región 3×3.

**"¿Por qué GoodsAD es tan difícil?"**
Las fotos no están alineadas y la única referencia suele ser otro producto u otra marca, así que una distancia entre parches marca todo el objeto. Lo mantenemos y lo reportamos aparte en lugar de quitarlo: quitarlo inflaría nuestros números.

**"¿Qué envía exactamente el experto a los lectores?"**
Una región 3×3, un zoom alrededor del pico y un puntaje. En las ablaciones también comparamos un mapa crudo, un mapa calibrado y una pista solo en texto.

**"¿Cómo van a demostrar que los lectores no simplemente copian al experto?"**
El caso de las chinches muestra que a veces lo hacen. La ablación "sin experto" mide el efecto neto, y el análisis de errores contará cuántas veces una pista equivocada llevó a una respuesta equivocada, con las categorías de MAST para los fallos de orquestación.

**"¿Por qué Ollama en local y vLLM en Colab?"**
vLLM no corre en Windows y 6 GB no alcanzan para un modelo de 8B en bf16. Ollama sirve un modelo 4B cuantizado con la misma API compatible con OpenAI, así que el código es idéntico; solo cambia la dirección del servidor.

**"¿18 imágenes alcanzan para decir algo?"**
Solo que el pipeline funciona: cero errores de formato en 184 llamadas, el verificador reabre sin entrar en bucle y los logs están completos. Las cifras de precisión no son un resultado, por eso la diapositiva muestra el intervalo de confianza [54 %, 86 %].

**"¿El test está realmente intacto?"**
Sí. Está fijado por una semilla commiteada, separado de dev, y la CLI rechaza `--split test` sin una configuración congelada. Se correrá una sola vez, después del 19 de noviembre.
