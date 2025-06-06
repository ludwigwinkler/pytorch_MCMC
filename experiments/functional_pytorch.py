import torch
import copy


inputs = torch.randn(64, 3)
targets = torch.randn(64, 3)
model = torch.nn.Linear(3, 3)

params = dict(model.named_parameters())


def compute_loss(params, inputs, targets):
    prediction = torch.func.functional_call(model, params, (inputs,))
    return torch.nn.functional.mse_loss(prediction, targets)


grads = torch.func.grad(compute_loss)(params, inputs, targets)


# %%

num_models = 5
batch_size = 64
in_features, out_features = 3, 3
models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
data = torch.randn(batch_size, 3)

# Construct a version of the model with no memory by putting the Tensors on
# the meta device.
base_model = copy.deepcopy(models[0])
base_model.to("mps")

params, buffers = torch.func.stack_module_state(models)


# It is possible to vmap directly over torch.func.functional_call,
# but wrapping it in a function makes it clearer what is going on.
def call_single_model(params, buffers, data):
    return torch.func.functional_call(base_model, (params, buffers), (data,))


def call_single_loss(params, buffers, data):
    prediction = torch.func.functional_call(base_model, (params, buffers), (data,))
    return torch.nn.functional.mse_loss(prediction, targets)


output = torch.vmap(call_single_model, (0, 0, None))(params, buffers, data)
"""
grad_and_value: outermost wrapper on around loss fucntion evaluation with input (parameters, buffers, data)
				argnums=(0,2) specififes the derivatives to be computed with respect to
torch.sum:		sum over multiple evaluations ([BS, params], [BS, buffers])
torch.vmap:		vectorized map over the first two arguments (params, buffers) while keeping data constant
"""
grad = torch.func.grad_and_value(
    lambda p, b, d: torch.sum(
        torch.vmap(call_single_loss, (0, 0, None))(p, b, d["data"])
    ),
    argnums=(0, 2),
)(params, buffers, {"data": data})
assert output.shape == (num_models, batch_size, out_features)
