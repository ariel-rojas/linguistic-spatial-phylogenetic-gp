"""Ejemplo mínimo para explorar la base unificada local de low."""

import low


def main():
    """Muestra el inventario y ejemplos de relaciones entre países y lenguas."""
    print("Cargando el grafo lingüístico (instantánea unificada incluida en low)...")

    # Inicializa la base de datos unificada
    db = low.LanguagesOfTheWorld()

    # 1. Imprime las estadísticas globales de los datos cargados
    print("\n--- Resumen Global ---")
    print(db)

    # 2. Accede a propiedades específicas (ej. cantidad de sistemas de escritura)
    print(f"\nSistemas de escritura registrados en el mundo: {len(db.scripts)}")
    lang = db.languages.get("spa")
    for country in lang.countries:
        print(f"{country.label} - {country.region.label} ({country.continent.label})")

    rw = db.countries.get("AR")
    for language in rw.languages:
        print(language.label)


if __name__ == "__main__":
    main()
