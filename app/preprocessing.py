def preprocess(data):
    """
    Preprocesa los datos enviados desde el formulario HTML antes de usarlos en el modelo.
    Convierte los datos de cadena de texto a números (float).
    """
    try:

        print(data)

        # Convierte las entradas del formulario a números flotantes
        return [
            float(data['overall_qual']),  # Calidad general
            float(data['gr_liv_area']),  # Área habitable
            float(data['garage_cars']),  # Número de coches en el garaje
            float(data['garage_area']),  # Área del garaje
            float(data['total_bsmt_sf']) # Área total del sótano
        ]
    except KeyError as e:
        raise ValueError(f"Falta un campo requerido en los datos del formulario: {e}")
    except ValueError as e:
        raise ValueError(f"Error al convertir los datos a números: {e}")
