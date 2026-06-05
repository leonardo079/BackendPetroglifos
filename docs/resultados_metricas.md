# Resultados de Métricas Técnicas

Este documento resume el comportamiento actual del sistema multiagente de petroglifos a partir de las corridas acumuladas en `storage/metrics/runs.jsonl` y del reporte autogenerado en `storage/metrics/report.json`.

## Estado Actual del Sistema

El flujo de trabajo está instrumentado para guardar resultados en cada corrida. Actualmente el sistema:

- registra cada ejecución en un archivo `JSONL`,
- recalcula automáticamente el reporte agregado,
- conserva el tiempo total de ejecución por caso,
- marca si hubo reconstrucción o no,
- guarda la imagen original, la preprocesada y la reconstruida cuando aplica.

En su estado actual, el sistema se comporta como un pipeline semi-automatizado de clasificación y documentación:

1. recibe una imagen,
2. la preprocesa,
3. detecta motivos,
4. decide si requiere reconstrucción,
5. reconstruye solo cuando lo considera necesario,
6. clasifica el petroglifo,
7. genera la ficha ICANH,
8. guarda la corrida para análisis posterior.

## Resumen Numérico

Con corte a la última revisión, el proyecto acumula:

| Métrica | Valor |
|---|---:|
| Corridas totales | 13 |
| Corridas exitosas | 13 |
| Tasa de éxito autónomo | 100% |
| Corridas con reconstrucción | 10 |
| Corridas sin reconstrucción | 3 |
| Tiempo medio de ficha | 0.82 min |
| Tiempo mediano de ficha | 0.83 min |
| Tiempo mínimo observado | 0.34 min |
| Tiempo máximo observado | 1.29 min |
| FID acumulado | 66.1237 |

## Métricas Técnicas Analizadas

### 1. Tasa de éxito autónomo

La tasa de éxito autónomo mide cuántas corridas completan el flujo completo sin intervención manual.

- Resultado actual: **13/13**
- Tasa: **100%**

Interpretación:

- El sistema está logrando completar el flujo completo en todas las corridas registradas.
- Esto indica que la orquestación, los agentes y la generación de fichas están funcionando de forma estable a nivel operativo.

### 2. Tiempo de producción de ficha por sitio

El tiempo se mide desde el inicio de la ejecución del pipeline hasta la generación de la ficha final.

- Promedio: **0.82 minutos**
- Mediana: **0.83 minutos**
- Rango observado: **0.34 a 1.29 minutos**
- Umbral objetivo del proyecto: **menor a 45 minutos**

Interpretación:

- El sistema supera ampliamente el objetivo planteado.
- Frente al proceso manual de semanas, la reducción es muy significativa.
- En términos prácticos, la automatización permite producir una ficha en menos de 2 minutos en el conjunto observado.

### 3. Reconstrucción GAN

La reconstrucción no se activa en todos los casos. El sistema decide reconstruir solo cuando detecta deterioro o cuando la calidad del caso lo sugiere.

- Corridas con reconstrucción: **10**
- Corridas sin reconstrucción: **3**

Interpretación:

- El comportamiento actual es coherente con un sistema adaptativo, no con una reconstrucción obligatoria para todas las imágenes.
- Esto es deseable, porque evita costear reconstrucciones innecesarias en petroglifos que ya están en buen estado.

### 4. FID de reconstrucciones

El FID evalúa la distancia entre las imágenes reconstruidas y las imágenes de referencia.

- FID actual: **66.1237**
- Objetivo del proyecto: **menor a 45**

Interpretación:

- Esta es la métrica que actualmente muestra más margen de mejora.
- El valor indica que las reconstrucciones aún están lejos del nivel de similitud visual deseado con el conjunto de referencia.
- A nivel de informe, esto puede presentarse como una oportunidad clara de optimización del modelo de reconstrucción o del posprocesamiento visual.

### 5. IoU de detección

La métrica de IoU todavía no se puede consolidar automáticamente porque no existen anotaciones ground truth cargadas en las corridas actuales.

- Estado actual: **omitida**
- Motivo: **no hay cajas de referencia disponibles**

Interpretación:

- El sistema ya está preparado para calcularla.
- Sin embargo, para reportarla correctamente necesita cajas anotadas por imagen.
- Puede incorporarse luego mediante un archivo JSON lateral o un conjunto de validación anotado.

## Cómo Está Funcionando Actualmente

En la práctica, el sistema opera así:

- **A1** normaliza y preprocesa la imagen.
- **A2** detecta motivos y estima si hay deterioro.
- **A5** se activa solo cuando hace falta reconstruir.
- **A3** compara iconográficamente con el corpus.
- **A4** genera análisis cultural con apoyo de RAG.
- **A6** produce la ficha final.
- El orquestador guarda todo en logs y archivos de métricas.

Desde el punto de vista de ingeniería, esto permite:

- rastrear cada corrida individual,
- acumular resultados sin separar imágenes manualmente,
- consolidar estadísticas al final del experimento,
- reutilizar los mismos logs para el informe de resultados.

## Oportunidades de Mejora

### 1. Incorporar ground truth para IoU

Para activar la métrica de detección de forma completa, conviene crear anotaciones de referencia por imagen.

Opciones recomendadas:

- archivo JSON lateral junto a cada imagen,
- archivo CSV de validación,
- formato COCO o similar si luego se quiere escalar.

### 2. Reducir el FID

El FID actual todavía supera la meta del proyecto. Algunas mejoras posibles:

- reentrenar el modelo de reconstrucción con más ejemplos del dominio rupestre,
- mejorar el balance entre preservación de textura y restauración,
- ajustar el posprocesamiento de salida,
- evaluar filtros de limpieza o refinamiento visual sobre la reconstrucción final.

### 3. Registrar más contexto por corrida

Aunque ya se guardan métricas útiles, se puede enriquecer el reporte con:

- confianza de detección,
- tipo de reconstrucción aplicada,
- número de motivos detectados,
- longitud del flujo de agentes activados,
- causa de reconstrucción o de omisión.

### 4. Separar mejor reconstrucción real vs. passthrough

En algunos casos el sistema marca el paso por reconstrucción aunque finalmente devuelve la imagen original. Para el informe conviene distinguir:

- reconstrucción real,
- paso sin reconstrucción,
- reconstrucción omitida por no requerirse.

Eso haría más clara la interpretación del número de casos reconstruidos.

## Lectura Para El Informe

Si se resume en una frase académica:

> El sistema muestra un desempeño operativo sólido en automatización y tiempos de respuesta, con 100% de éxito autónomo y tiempos de generación de ficha muy por debajo del umbral objetivo; sin embargo, el FID de reconstrucción sigue siendo el principal punto de mejora y la métrica IoU aún requiere un conjunto de referencia anotado para su validación formal.

## Conclusión Breve

El proyecto ya produce métricas acumuladas de forma automática y útil para el informe de resultados. Hoy puede defenderse con fuerza en:

- automatización,
- trazabilidad,
- eficiencia temporal,
- robustez operativa.

Las dos áreas que más conviene fortalecer son:

- evaluación formal de detección con IoU,
- calidad visual de las reconstrucciones para bajar el FID.
