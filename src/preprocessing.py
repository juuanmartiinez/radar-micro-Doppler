import numpy as np

def elim_offset(datos):
    return datos - np.mean(datos)

def resolucion_distancia(bw):
    return 3e8 / (2 * bw)

def fft_distancia(datos):
  return np.fft.fft(datos, axis=0)

def filtro_mti(perfil):
  return perfil - np.mean(perfil, axis=1, keepdims=True)

def detectar_rango(perfil_mti, margen_db=10, salto=2):
  
  salto_der = salto_izq = True
  
  potencia = np.abs(perfil_mti) ** 2
  
  energia = np.sum(potencia, axis=1)
  energia_db = 10 * np.log10(energia)
  umbral = np.median(energia_db) + margen_db

  semilla = np.argmax(energia_db)

  izquierda = semilla
  derecha = semilla

  while izquierda - 1 >= 0 and (energia_db[izquierda - 1] > umbral or salto_izq):

    if energia_db[izquierda - 1] < umbral:
      salto_izq = False
      izquierda = max(0, izquierda - salto)
    else:
      izquierda -= 1


  while derecha + 1 < len(energia_db) and (energia_db[derecha + 1] > umbral or salto_der):

    if energia_db[derecha + 1] < umbral:
      salto_der = False
      derecha = min(len(energia_db) - 1, derecha + salto)
    else:
      derecha += 1 

  return izquierda, derecha