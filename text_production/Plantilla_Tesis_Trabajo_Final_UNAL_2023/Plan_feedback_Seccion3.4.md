# Plan de implementación del feedback — Sección 3.4 "Modelo Federado"

**Archivo objetivo:** `03Seccion03.tex` (la Sección 3.4 ocupa las líneas **621–1003**)
**Fuente del feedback:** comentarios de `fagomezj` en `0000 (1)_FG.pdf`, págs. 29–33

> Nota de alcance: el PDF tiene comentarios tanto en **3.3 Métodos** (págs. 27–28) como en **3.4 Modelo Federado** (págs. 29–33). Este plan cubre únicamente **3.4**, como se pidió. Al final dejo la lista de los comentarios de 3.3 por si se quieren abordar después.
> Los comentarios son resaltados (*highlights*) sin coordenadas de texto exportables, así que el anclaje exacto a cada párrafo se infirió por página + tema. Donde hay incertidumbre lo marco con ⚠️.

---

## Resumen de los 12 comentarios de la sección 3.4

| # | Pág. | Comentario (resumen) | Subsección / líneas .tex |
|---|------|----------------------|--------------------------|
| C1 | 29 | Buscar una justificación del federado mejor adaptada al problema de herbarios (p. ej. falta de cómputo) | Intro 3.4, líneas 623–646 |
| C2 | 29 | "No veo que esta sea una justificación fuerte" + **texto sugerido completo** sobre variabilidad y *domain shift* | 3.4.1, líneas 660–677 |
| C3 | 29 | Replantear desde: variabilidad interinstitucional + generalización entre herbarios | 3.4.1, líneas 652–663 |
| C4 | 29 | "Esto es carreta", centrarlo en el problema que abordamos | Intro 3.4, párrafo genérico, líneas 636–646 |
| C5 | 30 | "Esto se llama *domain generalization*" | 3.4.1, líneas 673–677 ⚠️ |
| C6 | 31 | La figura es interesante, pero poner algo más conceptual que entiendan los biólogos | Figura 3.4.1, líneas 683–695 |
| C7 | 31 | "¿Esto se puede ilustrar con un dibujo?" | 3.4.2 Fases, líneas 698–760 ⚠️ |
| C8 | 31 | Pensar en algo más conceptual para representar YOLO | Caption / mención YOLOv8, línea 689 |
| C9 | 32 | "Esta fase no la entendí" | 3.4.2, **Fase 5 Criterio de parada**, líneas 746–760 |
| C10 | 32 | Si no se comprime, dejar solo el operador de compresión | 3.4.2, Fase 3, líneas 733–738 |
| C11 | 33 | Redactar todo en pasado | 3.4.3 (y revisar toda 3.4), líneas 763–874 |
| C12 | 33 | "Describir mejor esta parte" | 3.4.3, **Estrategia 1 y las estrategias en general**, líneas 775–874 |

---

## Acciones propuestas, por bloque

### Bloque A — Motivación del federado (C1, C2, C3, C4, C5)
Es el grupo más sustantivo: el revisor considera que la motivación actual es genérica ("carreta") y pide centrarla en el problema real de los herbarios.

1. **C4 — Recortar el párrafo genérico** (líneas 636–646: salud, banca, IoT, hospitales). Condensar a 1–2 frases y eliminar los ejemplos no relacionados con herbarios.
2. **C1 — Reescribir la introducción 3.4** (líneas 623–646) para motivar el federado desde restricciones propias de los herbarios: soberanía de datos institucionales, derechos de imagen, confidencialidad de localidades de especies amenazadas y **limitaciones de cómputo** de las instituciones para entrenar modelos propios (el revisor lo sugiere explícitamente).
3. **C2 + C3 — Reescribir la justificación de 3.4.1** (líneas 660–677). El revisor entregó un **texto sugerido completo** que conviene incorporar casi literal. Ejes:
   - Introducir el término/concepto **AIRs (Artefactos de Identificación y Referencia)**.
   - Argumentar la **variabilidad interinstitucional** (apariencia, dimensiones, ubicación, diseño) → problema de **cambio de dominio (*domain shift*)**.
   - Esquema: cada herbario entrena un detector local especializado → se agregan en un detector global robusto que generaliza a colecciones no vistas.
   - Sustituir la justificación débil actual (la del "detector con datos mezclados dominado por instituciones grandes").
4. **C5 — Nombrar el concepto**: donde se habla de generalizar a colecciones nuevas no observadas (líneas 673–677), añadir explícitamente que esto corresponde a **domain generalization**, con su cita.

> Texto sugerido por el revisor (a integrar en A3), verbatim para referencia:
> *"Aunque los AIRs cumplen funciones curatoriales similares… su apariencia visual, dimensiones, ubicación y diseño presentan una importante variabilidad entre instituciones. Esta heterogeneidad genera un problema de cambio de dominio… Por esta razón se propone un esquema federado donde cada herbario entrena inicialmente un detector especializado… Posteriormente, los modelos son agregados para construir un detector global capaz de reconocer Artefactos de Identificación y Reconocimiento de forma robusta… y generalizar a nuevas colecciones no observadas durante el entrenamiento."*

### Bloque B — Figuras conceptuales (C6, C7, C8)
El revisor pide ilustraciones más conceptuales, pensadas para una audiencia de biólogos.

5. **C6 — Rediseñar la Figura `federated_herbarium_strategy.jpg`** (líneas 683–695): versión conceptual que comunique "qué hacemos" sin jerga técnica. Mantener la idea (datos que no salen de cada herbario + backbone compartido), pero con lenguaje/iconografía accesible. *Entregable de figura nueva: pendiente de generar (TikZ o imagen).*
6. **C8 — Representación conceptual de YOLO** (línea 689 y donde se cite YOLO): sustituir/añadir un esquema conceptual de YOLO en vez de la mención técnica "YOLOv8".
7. **C7 — Añadir un diagrama del ciclo federado** ⚠️ en 3.4.2 (líneas 698–760): un dibujo que ilustre las fases (inicialización → entrenamiento local → comunicación → agregación → repetición). *Confirmar a qué párrafo exacto apunta el highlight.*

### Bloque C — Fases del entrenamiento (C9, C10)
8. **C10 — Simplificar la Fase 3 "Comunicación de parámetros"** (líneas 730–738): como en este trabajo no hay compresión ($\rho=1$), eliminar la discusión de cuantización y **dejar solo el operador de compresión definido** (o retirarlo si no aporta).
9. **C9 — Profundizar la Fase 5 "Criterio de parada"** (líneas 746–760): el revisor no entendió esta fase. Explicar con más detalle el criterio de convergencia: significado de $\mathcal{L}_t$ (pérdida global promediada de la ronda $t$), del umbral $\varepsilon$ y de $T_{\max}$; aclarar la lógica del *case* (continúa mientras la mejora entre rondas supere $\varepsilon$ y no se alcance el máximo de rondas) y, si aplica, los valores concretos usados en este trabajo.

### Bloque D — Estrategias de heterogeneidad (C11, C12)
10. **C11 — Pasar a tiempo pasado** toda la subsección 3.4.3 (líneas 763–874) y revisar el resto de 3.4 para consistencia (la tesis se redacta en pasado). Cambiar formas como "se presentan / consiste / permite" → "se presentaron / consistió / permitió".
11. **C12 — Describir mejor las estrategias** (líneas 775–874): el comentario apunta a la **Estrategia 1 "Espacio de clases unificado"** (líneas 775–788) y, en general, a las tres estrategias. Ampliar y clarificar cada una: para la Estrategia 1, explicar mejor cómo se colapsan las 11 etiquetas de Melbourne en los 6 grupos de UNAL y el efecto concreto de la pérdida de granularidad; revisar también las Estrategias 2 y 3 para nivelar profundidad y claridad de la descripción.

---

## Orden de ejecución sugerido
1. Bloque A (motivación) — es el cambio de fondo y condiciona el resto del relato.
2. Bloque C y D (texto: fases + pasado) — ediciones acotadas y de bajo riesgo.
3. Bloque B (figuras) — requiere generar arte nuevo; se deja al final.

## Verificación final
- Compilar el `.tex` y revisar que las referencias (`\ref`, `\cite`) y las figuras nuevas no rompan.
- Releer 3.4 completa para consistencia de tiempo verbal (pasado).
- Confirmar que se añadieron las citas faltantes (domain generalization, domain shift).
- Revisar los `\cite{}` vacíos existentes en el archivo.

---

## Apéndice — Comentarios de 3.3 Métodos (fuera de alcance, por si se abordan luego)
- p27: "Describir mejor con descripción matemática" (data augmentation / class weights).
- p27: "Ampliar la descripción matemática de la función de pérdida" (focal loss).
- p28: "¿Esto se hizo antes o después del data augmentation? …primero partir, luego aumentar" (riesgo de overfitting).
- p28: "Primero describir la partición y después la aumentación".
- p28: "¿Se refiere a píxeles VP? ¿a qué umbral? Incluir figura" (Precisión).
- p28: "Ilustrar en una imagen" (F1).
- p28: "¿Cómo se ve en detección de objetos? Figura ilustrativa".
- p29: "De nuevo nivel de píxeles, ¿cómo se ve en una imagen?" (IoU).
