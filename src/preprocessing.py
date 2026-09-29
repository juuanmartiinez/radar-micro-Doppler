import numpy as np

def elim_offset(datos):
    return datos - np.mean(datos)

def resolucion_distancia(bw):
    return 3e8 / (2 * bw)

def fft_distancia(datos):
  return np.fft.fft(datos, axis=0)

def filtro_mti(perfil):
  return perfil - np.mean(perfil, axis=1, keepdims=True)