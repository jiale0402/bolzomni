"""Check the actual GPU job environment; no downloads or model training."""
import importlib.metadata
import json
import platform
import subprocess
import sys


def main():
    report = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "packages": {}}
    for package in ("torch", "boltzgen", "pytorch-lightning"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    revision = subprocess.run(
        ["git", "-C", "vendor/boltzgen", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    report["source_revision"] = revision.stdout.strip() or None
    try:
        import torch
        import boltzgen
        report["boltzgen_import_path"] = boltzgen.__file__
        report["cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if not report["cuda_available"]:
            raise RuntimeError("CUDA is unavailable in this job environment")
        report["gpu"] = torch.cuda.get_device_name(0)
        report["vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
        x = torch.ones((32, 32), device="cuda", requires_grad=True)
        (x @ x).sum().backward()
        torch.cuda.synchronize()
        report["cuda_forward_backward"] = bool(torch.isfinite(x.grad).all())
        if not report["cuda_forward_backward"]:
            raise RuntimeError("CUDA gradient check failed")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(report, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
