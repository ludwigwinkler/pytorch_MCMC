I'm writing an MCMC (Markov Chain Monte Carlo) algorithm in PyTorch.

I want to heavily emphasize PyTorch torch.func methods in combination with torch.func.vmap to implement the MCMC algorithm efficiently.
The focus is on using functional programming paradigms and vectorization to create beautiful abstractions.

Always double check whether the code you write actually corresponds to the MCMC algorithm and is efficient in terms of PyTorch's functional programming capabilities.

When writing code, ensure that:
- You use `torch.func` methods where applicable.
- You utilize `torch.func.vmap` for vectorized operations where applicable.
- You maintain clarity and readability while adhering to functional programming principles.
- You avoid in-place operations to preserve functional purity.
- You document the code with clear comments explaining the functional abstractions used.
- You ensure that the code is efficient and leverages PyTorch's capabilities for MCMC algorithms