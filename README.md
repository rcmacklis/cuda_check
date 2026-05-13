# CudaCheck

Script minimo para verificar si PyTorch detecta CUDA, listar las GPUs disponibles en tabla y ejecutar una suma simple en cada GPU.

## Requisitos

- Linux con driver NVIDIA funcional.
- Python 3.10.20.
- `uv` para crear el entorno virtual y gestionar paquetes.

## Instalacion

Instalar `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reiniciar la shell o cargar el binario de `uv` segun indique el instalador.

Crear el entorno virtual con Python 3.10.20:

```bash
uv venv --python 3.10.20 --seed
```

Activar el entorno:

```bash
source .venv/bin/activate
```

Instalar PyTorch con CUDA 13:

```bash
uv pip install -r pyproject.toml --index-url https://download.pytorch.org/whl/cu130
```

Si tu GPU o driver requiere otra version CUDA, cambia el indice por el recomendado en el selector oficial de PyTorch, por ejemplo `cu128`, `cu126` o `cu118`.

## Uso

Ejecutar la comprobacion completa:

```bash
python cuda_check.py
```

Comprobacion rapida:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Codigos de salida:

- `0`: CUDA disponible y prueba OK en todas las GPUs detectadas.
- `1`: PyTorch no esta instalado.
- `2`: CUDA no esta disponible para PyTorch.
- `3`: CUDA esta disponible, pero fallo la prueba en una GPU.

## Salida de ejemplo

```text
--- Entorno ---
Python: 3.10.20
PyTorch: 2.12.0+cu130
CUDA en PyTorch: 13.0
PyTorch compilado con CUDA: Si

--- Verificacion CUDA ---
GPUs detectadas: 1

GPU | Nombre                    | Compute | Memoria   | Prueba | Resultado      
----+---------------------------+---------+-----------+--------+----------------
0   | NVIDIA GeForce RTX 5070 Ti | 12.0    | 15.54 GiB | OK     | [5.0, 7.0, 9.0]

CUDA esta disponible y PyTorch puede ejecutar operaciones en todas las GPUs detectadas.
```
