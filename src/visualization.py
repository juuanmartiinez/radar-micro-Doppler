import numpy as np
import matplotlib.pyplot as plt

from .preprocessing import resolucion_distancia, energia_casillas_db

def _ejes(perfil_mti, params):
    res = resolucion_distancia(params["bw"])
    eje_dist = np.arange(perfil_mti.shape[0]) * res
    eje_t = np.arange(perfil_mti.shape[1]) * params["t_chirp"] / 1000
    
    return res, eje_dist, eje_t


def _nuevo_ax(ax, figsize):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    
    return ax


def plot_mapa_distancia(perfil_mti, params, rango=None, rango_db=40,
                        titulo=None, ax=None):
    ax = _nuevo_ax(ax, (10, 5))
    res, eje_dist, eje_t = _ejes(perfil_mti, params)

    perfil_db = 20 * np.log10(np.abs(perfil_mti))
    vmax = perfil_db.max()
    im = ax.imshow(perfil_db, aspect="auto", origin="lower",
                   extent=[0, eje_t[-1], 0, eje_dist[-1]],
                   vmin=vmax - rango_db, vmax=vmax)
    if rango is not None:
        for casilla in rango:
            ax.axhline(casilla * res, color="orange", linestyle="--")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Distancia (m)")
    ax.set_title(titulo or "Mapa distancia-tiempo (MTI)")
    plt.colorbar(im, ax=ax, label="dB")
    
    return ax


def plot_energia_casillas(perfil_mti, params, rango=None, margen_db=10,
                          titulo=None, ax=None):
    ax = _nuevo_ax(ax, (10, 4))
    res, eje_dist, _ = _ejes(perfil_mti, params)

    energia_db = energia_casillas_db(perfil_mti)
    umbral = np.median(energia_db) + margen_db

    ax.plot(eje_dist, energia_db)
    ax.axvline(np.argmax(energia_db) * res, color="red", linestyle="--", label="argmax")
    ax.axhline(umbral, color="green", linestyle="--", label=f"mediana + {margen_db} dB")
    if rango is not None:
        ax.axvline(rango[0] * res, color="orange", label="rango")
        ax.axvline(rango[1] * res, color="orange")
    ax.set_xlabel("Distancia (m)")
    ax.set_ylabel("Energía (dB)")
    ax.set_title(titulo or "Energía por casilla")
    ax.legend()
    ax.grid(True)
    
    return ax


def plot_senal_temporal(senal, params, titulo=None, ax=None):
    ax = _nuevo_ax(ax, (10, 4))
    eje_t = np.arange(len(senal)) * params["t_chirp"] / 1000

    ax.plot(eje_t, 20 * np.log10(np.abs(senal)))
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("|señal| (dB)")
    ax.set_title(titulo or "Señal en tiempo lento")
    ax.grid(True)
    
    return ax


def plot_resumen(perfil_mti, params, rango, senal, titulo=""):
    fig, axs = plt.subplots(3, 1, figsize=(10, 11))
    plot_mapa_distancia(perfil_mti, params, rango=rango, ax=axs[0],
                        titulo=f"{titulo} - mapa distancia-tiempo")
    plot_energia_casillas(perfil_mti, params, rango=rango, ax=axs[1],
                          titulo=f"{titulo} - energía por casilla")
    plot_senal_temporal(senal, params, ax=axs[2],
                        titulo=f"{titulo} - señal en tiempo lento")
    fig.tight_layout()
    
    return fig