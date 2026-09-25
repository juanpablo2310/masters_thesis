# Entrenar el modelo federado en Colab (GPU)

Tu Mac tiene poca RAM y problemas de permisos con el disco externo. Colab con GPU
entrena esto en minutos y sin esos dolores de cabeza.

## Paso 1 — Comprimir los datos

En tu Mac (Terminal, con el disco externo accesible en Finder):

```bash
cd "/Volumes/ADATA HD680/Shared/Files From d.localized/Maestria/tesis/herbario"
zip -r ~/Desktop/data_UN.zip   data_UN
zip -r ~/Desktop/data_MELU.zip data_MELU
```

Cada zip debe contener, adentro, `.../train/images` y `.../train/labels`.
(Si el disco te da "Operation not permitted" también en Terminal, comprime desde
**Finder**: clic derecho en la carpeta → "Comprimir".)

## Paso 2 — Subir a Google Drive

Crea en tu Drive la carpeta `MyDrive/herbario/` y sube ahí:

- `data_UN.zip`
- `data_MELU.zip`
- La carpeta **`FederatedModel`** completa (los `.py`: `class_mask.py`, `client.py`,
  `server.py`, `Trainer.py`, `metrics.py`, `VisualizationTools.py`, `AnaliticTools.py`,
  `privacy.py`, `model_comparison.py`).

Debe quedar así:

```
MyDrive/herbario/
├── FederatedModel/         (los .py)
├── data_UN.zip
└── data_MELU.zip
```

## Paso 3 — Abrir el notebook en Colab

1. Ve a https://colab.research.google.com → **Archivo → Subir cuaderno** →
   sube `herbario_federated_colab.ipynb`.
2. **Entorno de ejecución → Cambiar tipo de entorno → T4 GPU**.
3. Corre las celdas en orden. En la celda 3 solo confirma que las 4 rutas coinciden
   con dónde subiste las cosas (si usaste `MyDrive/herbario/` no hay que cambiar nada).

## Qué esperar

- La celda 8 entrena (FedAvg sincronizado, máscara por institución). Cada 5 rondas
  imprime el `mAP50` global — **debe subir** ronda a ronda.
- El modelo final (`global_model_round_*.pt`) se guarda en
  `MyDrive/herbario/federated_output/` (persiste en Drive).
- La celda 9 confirma que detecta **ambas** instituciones (UNAL 0–5 y Melbourne 6–16).

## Notas

- Si el `mAP50` sigue subiendo en la ronda 40, sube `rounds` (p.ej. 80) y re-corre.
- El código ya es agnóstico al entorno: la única ruta hardcodeada (dónde ultralytics
  escribe los runs) se controla con `os.environ['HERBARIO_RUNS_DIR']`, que el notebook
  fija en `/content/runs`. Nada que editar en los `.py`.
- Para el comparativo por clase completo, corre `model_comparison.py` con el checkpoint
  federado resultante como `--federated`.
