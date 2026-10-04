import numpy as np
import os, re

ACTIVIDADES = {1: "andar", 2: "sentarse", 3: "levantarse",
               4: "agacharse", 5: "beber", 6: "caerse"}

# Errores de etiquetado detectados en el dataset original
CORRECCIONES = {
    "C5_6P01A05R03": 5,   # el nombre dice caerse, pero el espectrograma es beber
}

def cargar_dat(ruta):
    with open(ruta) as f:
        lineas = f.readlines()

        params = {
            "fc":   float(lineas[0]),
            "t_chirp":    float(lineas[1]),
            "n_muestras": int(lineas[2]),
            "bw":         float(lineas[3])
        }

    datos = []

    for complejo in lineas[4:]:
        complejo = complejo.strip()
        complejo = complejo.replace("i", "j")
        complejo = complex(complejo)
        datos.append(complejo)
    
    datos = np.array(datos)

    n = params["n_muestras"]
    assert len(datos) % n == 0, f"{ruta}: {len(datos)} datos, no es múltiplo de {n}"
    datos = datos.reshape((n, -1), order="F")

    return params, datos

_PATRON = re.compile(r"(\d)P(\d+)A(\d+)R(\d+)", re.IGNORECASE)

def parsear_nombre(ruta):
    nombre = os.path.basename(ruta)
    m = _PATRON.match(nombre)
    if m is None:
        raise ValueError(f"Nombre no reconocido: {nombre}")
    actividad, persona, _, repeticion = map(int, m.groups())
    return {
        "actividad": actividad,
        "persona": persona,
        "repeticion": repeticion,
        "campana": os.path.basename(os.path.dirname(ruta)),
    }