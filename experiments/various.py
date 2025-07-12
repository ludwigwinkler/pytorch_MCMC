import torch
from tensordict import TensorDict, NonTensorData
from collections.abc import Iterable

# %%
td = TensorDict(
    {"a": torch.randn(100, 4), "b": torch.randn(100, 4), "c": "abc", "d": 1.0}
)
td_sub = td.select("a", "b")
td.select("x")

# %%


class InfiniteLoader:
    def __init__(self, batch_size, shuffle=True):
        self.dataset = torch.randn(100, 4)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = self.dataset.shape[0]

    def get_batch(self, batch_size):
        idx = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        batch_idx = idx[: self.batch_size]
        return self.dataset[batch_idx]

    def __iter__(self):
        while True:
            yield self.get_batch(self.batch_size)


# Example usage:
loader = InfiniteLoader(batch_size=16, shuffle=True)

print(f"Is InfiniteLoader iterable? {isinstance(loader, Iterable)}")

td = TensorDict({"data": loader})
# td = TensorDict({'a': torch.ones(100, 4)})

# print(next(iter(NonTensorData(loader).data)))

# %%
for _ in range(2):
    data_sample = td.apply(
        lambda x, td: x.data.get_batch() if type(td) is NonTensorData else x, td
    )
    # print(td)
    print(data_sample["data"].std())
