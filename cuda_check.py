import platform
import sys


def load_torch():
    try:
        import torch
    except ImportError:
        print("[ERROR] El modulo 'torch' no se encontro.")
        print("Instala las dependencias con: uv pip install -r pyproject.toml --index-url https://download.pytorch.org/whl/cu130")
        sys.exit(1)

    return torch


def format_gib(value):
    return f"{value / (1024 ** 3):.2f} GiB"


def print_environment(torch):
    cuda_version = torch.version.cuda or "No disponible"
    cuda_built = "Si" if torch.backends.cuda.is_built() else "No"

    print("--- Entorno ---")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA en PyTorch: {cuda_version}")
    print(f"PyTorch compilado con CUDA: {cuda_built}")


def test_gpu(torch, index):
    device = torch.device(f"cuda:{index}")
    props = torch.cuda.get_device_properties(index)
    capability = f"{props.major}.{props.minor}"

    try:
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)
        result = (a + b).cpu().tolist()
        status = "OK"
        error = ""
    except Exception as exc:
        result = "-"
        status = "FALLO"
        error = str(exc)

    return {
        "GPU": str(index),
        "Nombre": props.name,
        "Compute": capability,
        "Memoria": format_gib(props.total_memory),
        "Prueba": status,
        "Resultado": str(result),
        "Error": error,
    }


def print_table(rows):
    columns = ["GPU", "Nombre", "Compute", "Memoria", "Prueba", "Resultado"]
    widths = {
        column: max(len(column), *(len(row[column]) for row in rows))
        for column in columns
    }

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)

    print(header)
    print(separator)
    for row in rows:
        print(" | ".join(row[column].ljust(widths[column]) for column in columns))


def main():
    torch = load_torch()

    print_environment(torch)
    print("\n--- Verificacion CUDA ---")

    if not torch.cuda.is_available():
        print("CUDA NO esta disponible. PyTorch no puede acceder a la GPU.")
        print("Revisa el driver NVIDIA y que PyTorch se haya instalado con un wheel CUDA.")
        return 2

    gpu_count = torch.cuda.device_count()
    print(f"GPUs detectadas: {gpu_count}\n")

    rows = [test_gpu(torch, index) for index in range(gpu_count)]
    print_table(rows)

    failed_rows = [row for row in rows if row["Prueba"] != "OK"]
    if failed_rows:
        print("\n--- Errores ---")
        for row in failed_rows:
            print(f"GPU {row['GPU']}: {row['Error']}")
        return 3

    print("\nCUDA esta disponible y PyTorch puede ejecutar operaciones en todas las GPUs detectadas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
