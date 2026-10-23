# Arquitectura del visualizador

```text
Release → prepare.py → tgz originales + catalog.json
MCC original ───────┐
Catálogo y tgz ─────┼→ readers.py → árbol plano + metadata por nodo
WALS preparado ────┘                    ↓
                             app.py: API, caché de tres árboles
                                        ↓
                    app.js: controles y fichas · tree.js: dibujo SVG
```

Se separan adquisición, lectura, servicio, controles y dibujo. No se introduce una
clase abstracta para dos formatos: los lectores son funciones con esquema común.
El visualizador no importa modelos del GP ni modifica sus artefactos.

## Esquema

Cada árbol tiene `entry` y una lista plana `nodes`. Cada nodo incluye `id`, `parent`,
`children`, `label`, `source_label`, `glottocode`, `leaves`, `length`, `depth` y `age_years`.
Los ID son locales. Longitud y profundidad conservan milenios. La edad deriva de la
profundidad máxima y el momento del corte; se muestra en años.

La metadata se conserva por nodo: anotaciones, anotaciones de rama, comentarios,
fila CSV y lengua/WALS. Se preservan propiedades desconocidas y comentarios literales.
No se incluye toda esa metadata en la respuesta de geometría.

## API de lectura

- `/api/catalog`: combinaciones disponibles y hashes de fuentes.
- `/api/trees/{id}`: topología, nombres y tiempos.
- `/api/trees/{id}/nodes/{node_id}`: todas las propiedades de un nodo.

Solo se aceptan ID del catálogo, nunca rutas enviadas por el navegador. El servidor
escucha en 127.0.0.1 y los valores externos se insertan como texto en la interfaz.

## Preservación y validación

El MCC tiene nombres sin escapar en Taxlabels: se extraen Translate y el Newick
anotado STATE_1, que DendroPy interpreta sin eliminar comentarios. STATE_0 es una copia
sin anotaciones, no otra muestra. Los originales permanecen intactos.

Los tgz se leen por offset descomprimido. Su orden no identifica muestras. La preparación
verifica hashes y exige árbol y tabla por corte. La carga valida correspondencia exacta
de puntas y filas, número de muestra, fecha, Crown−Time y ramas no negativas.

## Navegación

Cuatro niveles abiertos al inicio; la búsqueda abre ancestros hasta el nodo. El eje
horizontal usa tiempos de ramas; el vertical organiza nodos visibles. Colapsar no
modifica el árbol, las edades ni los recuentos.

Entre árboles actuales se busca el MRCA del conjunto de glottocodes descendientes
seleccionado. Si incorpora puntas adicionales se informa. Los números de los cortes
son ID locales: todavía no reconstruimos correspondencias con hojas modernas y la
interfaz avisa cuando vuelve a la raíz.

Solo se ofrecen fechas de árboles publicados. Cortar en cualquier año requeriría
implementar y validar una nueva poda, con decisiones sobre metadata ancestral.
No se presenta interpolación como si fuera un dato de la release.
