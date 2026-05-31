from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Subset


def extract_embeddings(
    model: torch.nn.Module,
    dataset,
    device: str | torch.device,
    batch_size: int = 128,
    indices: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference and return embeddings and labels on CPU.

    If ``indices`` is provided, only those dataset items are embedded.
    Dataset order is preserved because the dataloader never shuffles.
    """
    model.eval()

    eval_dataset = Subset(dataset, list(indices)) if indices is not None else dataset
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            batch_embeddings = model.forward_embedding(x)
            embeddings.append(batch_embeddings.cpu())
            labels.append(y.cpu())

    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)
