# Lectura de languages-of-the-world

Revisión para este experimento, 5 de octubre de 2026. La implementación usada es la
instalada en el entorno del proyecto, versión 0.2.0, fijada por `uv.lock`.
Referencia: [repositorio de jnehring](https://github.com/jnehring/languages-of-the-world).
El hash de su base incorporada se guarda en `datos/fuentes.json`.

## Qué sucede al crear la base

`low.LanguagesOfTheWorld()` lee `low/data/low_db.json` y construye objetos Python en
memoria. No consulta siete servidores en cada llamada. `graph.py` primero crea
continentes, regiones y países; luego crea nodos familiares, conecta padres e hijos
y finalmente vincula lenguas, nombres, escrituras y estimaciones de hablantes.
Las relaciones incluyen referencias de vuelta: una lengua conoce sus países y
un país permite consultar sus lenguas.

`models.py` define las entidades con dataclasses. `collections.py` proporciona
iteración, longitud, índices, cortes y búsquedas mediante diccionarios. No es un
motor de consultas ni una base relacional persistente. Aunque la API se describe
como de lectura, los objetos Python son mutables: no modificamos sus relaciones.

## Qué conecta y de dónde viene

| Fuente | Aporte utilizado por la biblioteca |
|---|---|
| SIL ISO 639-3 | Inventario base, código principal, nombre y alcance individual/macrolengua |
| UN M49 | Países, regiones y continentes |
| LinguaMeta | Países de uso, nombres, escrituras y estimaciones de hablantes |
| Glottolog CLDF | Glottocode, clasificación jerárquica y estado de amenaza |
| Unicode CLDR | Poblaciones, hablantes por territorio, oficialidad y nombres de escrituras |
| CIA World Factbook | Estimaciones de hablantes por país |
| Wikidata | Estimaciones globales de hablantes |

La reconstrucción vive en `bootstrap.py`: descarga fuentes, interpreta formatos,
normaliza identificadores y genera el JSON empaquetado. Puede incorporar además
una instantánea de recuentos del scraper. Varias descargas accesorias capturan
errores y continúan con listas vacías; una reconstrucción que termina no demuestra
por sí sola que estén presentes todas las fuentes. No ejecutamos bootstrap.

Los países vinculados a una lengua y los recuentos por país proceden de rutas
distintas de integración. No se deben confundir país de uso, país de origen y
localización de una variedad. Los hablantes tampoco son una medida homogénea:
el total global se combina tomando el máximo entre LinguaMeta y Wikidata; los
recuentos por país se conservan por fuente y se deduplican tomando máximos dentro
de cada combinación país/lengua/fuente. No son observaciones independientes
para nuestro GP ni se suman indiscriminadamente.

## Cómo navegarla

```python
import low

db = low.LanguagesOfTheWorld()
language = db.languages.get("spa")
print(language.label, language.glottocode)

node = language.family
while node is not None:
    print(node.glottocode, node.label)
    node = node.parent
```

`db.languages.get()` distingue códigos ISO de dos o tres caracteres y, para
otras longitudes, busca una etiqueta. **No busca lenguas por glottocode**. Por eso
nuestro código construye explícitamente un índice de glottocodes y comprueba
unicidad. `db.families.get()` sí permite buscar nodos familiares por glottocode.
Una búsqueda sin resultado devuelve `None`; los nombres no garantizan unicidad.

`lang.family` es el nodo familiar inmediato, no necesariamente la familia raíz.
`family.root` recorre hasta la raíz; `family.ancestors` excluye el propio nodo.
`family.languages` incluye solo lenguas cuyo padre inmediato es ese nodo, no todos
sus descendientes. Las propiedades `children`, `parent` y `depth` describen la
jerarquía, pero no longitudes de rama ni tiempos de divergencia.

Al integrar Glottolog, bootstrap conserva nodos de nivel familiar y asigna glottocodes
a registros de nivel lengua que tengan ISO. Si el padre directo no es un nodo
familiar retenido, asciende hasta encontrar uno. Por tanto low no expone todo el
universo de variedades de Glottolog. Las raíces administrativas también se incorporan;
hay que distinguirlas de las familias genealógicas para construir una covarianza.

## Consecuencias para nuestro proyecto

Usamos low como puente de identificadores y metadatos de familia; WALS aporta las unidades,
coordenadas y respuesta tipológica. No usamos nombres, número de hablantes ni país
como sustitutos de coordenadas o genealogía. La clasificación puede diferir entre
versiones de WALS y Glottolog, por lo que conservamos `Family` de WALS y `familia_low`
como campos diferentes. No mezclamos ambas clasificaciones para rellenar huecos.

La ausencia de familia puede representar un aislado, un enlace incompleto o datos ausentes. Ya no excluye del GP: la covarianza procede exclusivamente del árbol MCC y basta una hoja enlazada, coordenadas válidas e identidad sin conflicto. Las familias ausentes sí se excluyen de la evaluación por familias y quedan registradas; no se agrupan artificialmente como una familia.

La base distribuida no equivale a una actualización en vivo ni fija necesariamente
las revisiones originales de cada fuente ascendente. Fijar el paquete y su hash
permite identificar lo usado; no inventamos una versión Glottolog a partir de ello.
El código del repositorio usa licencia MIT; las fuentes integradas mantienen sus
propias condiciones. Los datos WALS utilizados declaran CC BY 4.0 en sus metadatos.
