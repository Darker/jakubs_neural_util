
from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn as nn

class CoreTorchModule(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.external_submodules: dict[str, nn.Module] = {}

    @abstractmethod
    def get_base_filename(self) -> str:
        """
        Must return the default base name for weights of this module
        """
        pass

    def save_weights(self, target_dir: Path, *, singlefile=False):
        base = self.get_base_filename()

        if singlefile:
            torch.save(self.state_dict(), target_dir / f"{base}_all.weights")
            return

        # Save core (excluding external submodules)
        core_state = {
            k: v for k, v in self.state_dict().items()
            if not any(k.startswith(name + ".") for name in self.external_submodules)
        }
        torch.save(core_state, target_dir / f"{base}.weights")

        # Save each external submodule
        for name, module in self.external_submodules.items():
            torch.save(
                module.state_dict(),
                target_dir / f"{base}.{name}.weights"
            )

    def load_weights(self, target_dir: Path, *, require_exists=False, singlefile=False):
        base = self.get_base_filename()

        if singlefile:
            path = target_dir / f"{base}_all.weights"
            if not path.exists():
                raise FileNotFoundError(path)
            self.load_state_dict(torch.load(path, weights_only=True))
            return (1, 1)

        found = 0
        needed = 1 + len(self.external_submodules)

        # Load core
        core_path = target_dir / f"{base}.weights"
        if core_path.exists():
            core_state = torch.load(core_path, weights_only=True)
            self.load_state_dict(core_state, strict=False)
            found += 1
        elif require_exists:
            raise FileNotFoundError(core_path)

        # Load submodules
        for name, module in self.external_submodules.items():
            path = target_dir / f"{base}.{name}.weights"
            if path.exists():
                module.load_state_dict(torch.load(path, weights_only=True))
                found += 1
            elif require_exists:
                raise FileNotFoundError(path)
            else:
                print(f"[Warning] Missing weights for {name}: {path}")

        return (found, needed)
