import numpy as np

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