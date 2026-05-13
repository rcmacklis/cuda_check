# AGENTS.md

## Repo
- Proyecto minimo de Python: `cuda_check.py` verifica si PyTorch detecta CUDA, lista GPUs y ejecuta una suma simple en cada GPU.
- Usa `uv` para crear `.venv` y gestionar paquetes; no hay tests, lint ni CI configurados.
- Trata `.venv/` como entorno local generado; no edites paquetes dentro de `.venv/`.

## Setup
- Instalar `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Reiniciar la shell o cargar el binario de `uv` segun indique el instalador.
- Crear entorno con Python 3.10.20 y paquetes seed: `uv venv --python 3.10.20 --seed`
- Activar entorno: `source .venv/bin/activate`
- Instalar dependencias con CUDA 13: `uv pip install -r pyproject.toml --index-url https://download.pytorch.org/whl/cu130`
- Si la maquina requiere otra version CUDA, reemplazar el indice por `cu128`, `cu126` o `cu118` segun el selector oficial de PyTorch.

## Run
- Ejecutar verificacion: `python cuda_check.py`
- Verificacion rapida: `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`

## Gotchas
- `torch.cuda.is_available()` depende del driver NVIDIA y de instalar un wheel CUDA de PyTorch; instalar `torch` sin el indice CUDA puede dejar CUDA no disponible.
- Codigos de salida de `cuda_check.py`: `0` OK, `1` falta PyTorch, `2` CUDA no disponible, `3` fallo la prueba en al menos una GPU.
