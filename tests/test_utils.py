import pytest
import torch
import random
import string
from tensordict import TensorDict
from collections import OrderedDict


def generate_random_string(length: int = 10) -> str:
    """Generate a random string of specified length or random length between 3-10."""
    if length is None:
        length = random.randint(3, 10)

    # Use letters and numbers for valid Python identifiers
    chars = string.ascii_letters + string.digits + "_"
    # Ensure first character is a letter or underscore
    first_char = random.choice(string.ascii_letters + "_")
    rest_chars = "".join(random.choices(chars, k=length - 1))
    return first_char + rest_chars


def generate_random_tensor() -> torch.Tensor:
    """Generate a random tensor with random shape and values."""
    # Random number of dimensions (1-4)
    ndim = random.randint(1, 4)

    # Random shape for each dimension (1-10)
    shape = [random.randint(1, 10) for _ in range(ndim)]

    # Random tensor type
    tensor_type = random.choice(["float", "int", "bool"])

    if tensor_type == "float":
        return torch.randn(*shape)
    elif tensor_type == "int":
        return torch.randint(0, 100, shape)
    else:  # bool
        return torch.randint(0, 2, shape, dtype=torch.bool)


class TestTensorDictKeyOrder:
    """Test suite for TensorDict key order preservation."""

    def test_three_entry_dict_key_order_preservation(self):
        """Test that TensorDict preserves key order for a 3-entry dictionary."""
        # Generate 3 unique random keys
        keys = set()
        while len(keys) < 3:
            keys.add(generate_random_string())
        keys = list(keys)

        # Create dictionary with random tensors
        original_dict = {
            keys[0]: generate_random_tensor(),
            keys[1]: generate_random_tensor(),
            keys[2]: generate_random_tensor(),
        }

        # Convert to TensorDict
        tensor_dict = TensorDict(original_dict)

        # Extract keys from both
        original_keys = list(original_dict.keys())
        tensordict_keys = list(tensor_dict.keys())

        # Assert key order is preserved
        assert original_keys == tensordict_keys, (
            f"Key order not preserved!\n"
            f"Original: {original_keys}\n"
            f"TensorDict: {tensordict_keys}"
        )

        # Also verify all keys are present
        assert set(original_keys) == set(tensordict_keys), (
            f"Key sets don't match!\n"
            f"Original: {set(original_keys)}\n"
            f"TensorDict: {set(tensordict_keys)}"
        )

    def test_multiple_random_dicts_key_order(self):
        """Test key order preservation across multiple random dictionaries."""
        num_tests = 50  # Test many random cases

        for i in range(num_tests):
            # Generate 3 unique random keys
            keys = set()
            while len(keys) < 3:
                keys.add(generate_random_string())
            keys = list(keys)

            # Randomly shuffle the keys to test different orders
            random.shuffle(keys)

            # Create dictionary with random tensors
            original_dict = {
                keys[0]: generate_random_tensor(),
                keys[1]: generate_random_tensor(),
                keys[2]: generate_random_tensor(),
            }

            # Convert to TensorDict
            tensor_dict = TensorDict(original_dict)

            # Check key order preservation
            original_keys = list(original_dict.keys())
            tensordict_keys = list(tensor_dict.keys())

            assert original_keys == tensordict_keys, (
                f"Test {i + 1}/{num_tests} failed!\n"
                f"Key order not preserved!\n"
                f"Original: {original_keys}\n"
                f"TensorDict: {tensordict_keys}"
            )

    def test_ordered_dict_key_order(self):
        """Test that TensorDict preserves OrderedDict key order."""
        # Generate random keys
        keys = set()
        while len(keys) < 3:
            keys.add(generate_random_string())
        keys = list(keys)

        # Create OrderedDict with specific order
        ordered_dict = OrderedDict()
        # Add keys in a specific order
        for key in keys:
            ordered_dict[key] = generate_random_tensor()

        # Convert to TensorDict
        tensor_dict = TensorDict(ordered_dict)

        # Check key order preservation
        original_keys = list(ordered_dict.keys())
        tensordict_keys = list(tensor_dict.keys())

        assert original_keys == tensordict_keys, (
            f"OrderedDict key order not preserved!\n"
            f"Original: {original_keys}\n"
            f"TensorDict: {tensordict_keys}"
        )

    def test_key_order_with_different_tensor_types(self):
        """Test key order preservation with different tensor types."""
        # Generate unique keys
        keys = [generate_random_string() for _ in range(3)]
        while len(set(keys)) < 3:  # Ensure uniqueness
            keys = [generate_random_string() for _ in range(3)]

        # Create dict with different tensor types
        original_dict = {
            keys[0]: torch.randn(5, 3),  # float tensor
            keys[1]: torch.randint(0, 10, (4, 2)),  # int tensor
            keys[2]: torch.randint(0, 2, (3, 3), dtype=torch.bool),  # bool tensor
        }

        # Convert to TensorDict
        tensor_dict = TensorDict(original_dict)

        # Check key order and tensor values
        original_keys = list(original_dict.keys())
        tensordict_keys = list(tensor_dict.keys())

        assert original_keys == tensordict_keys, (
            f"Key order not preserved with different tensor types!\n"
            f"Original: {original_keys}\n"
            f"TensorDict: {tensordict_keys}"
        )

        # Also verify tensor values are preserved
        for key in original_keys:
            assert torch.equal(
                original_dict[key], tensor_dict[key]
            ), f"Tensor values not preserved for key '{key}'"

    def test_key_order_with_nested_structure(self):
        """Test key order preservation with batch dimensions."""
        keys = [generate_random_string() for _ in range(3)]
        while len(set(keys)) < 3:
            keys = [generate_random_string() for _ in range(3)]

        # Create dict with tensors that have batch dimensions
        batch_size = random.randint(2, 5)
        original_dict = {
            keys[0]: torch.randn(batch_size, 5),
            keys[1]: torch.randn(batch_size, 3, 2),
            keys[2]: torch.randn(batch_size, 1),
        }

        # Convert to TensorDict with batch_size
        tensor_dict = TensorDict(original_dict, batch_size=[batch_size])

        # Check key order preservation
        original_keys = list(original_dict.keys())
        tensordict_keys = list(tensor_dict.keys())

        assert original_keys == tensordict_keys, (
            f"Key order not preserved with batch dimensions!\n"
            f"Original: {original_keys}\n"
            f"TensorDict: {tensordict_keys}\n"
            f"Batch size: {batch_size}"
        )

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 999])
    def test_reproducible_key_order(self, seed):
        """Test key order preservation with different random seeds."""
        random.seed(seed)
        torch.manual_seed(seed)

        # Generate reproducible random dictionary
        keys = [generate_random_string() for _ in range(3)]
        while len(set(keys)) < 3:
            keys = [generate_random_string() for _ in range(3)]

        original_dict = {
            keys[0]: generate_random_tensor(),
            keys[1]: generate_random_tensor(),
            keys[2]: generate_random_tensor(),
        }

        # Convert to TensorDict
        tensor_dict = TensorDict(original_dict)

        # Check key order preservation
        original_keys = list(original_dict.keys())
        tensordict_keys = list(tensor_dict.keys())

        assert original_keys == tensordict_keys, (
            f"Key order not preserved with seed {seed}!\n"
            f"Original: {original_keys}\n"
            f"TensorDict: {tensordict_keys}"
        )


if __name__ == "__main__":
    # Run a quick test
    test_instance = TestTensorDictKeyOrder()

    print("Running quick key order test...")
    test_instance.test_three_entry_dict_key_order_preservation()
    print("✓ Three entry dict test passed!")

    test_instance.test_multiple_random_dicts_key_order()
    print("✓ Multiple random dicts test passed!")

    test_instance.test_key_order_with_different_tensor_types()
    print("✓ Different tensor types test passed!")

    print("\nAll tests passed! TensorDict preserves key order correctly.")

    # Example of a test run
    print("\n" + "=" * 50)
    print("Example test run:")

    # Generate example
    keys = [generate_random_string() for _ in range(3)]
    while len(set(keys)) < 3:
        keys = [generate_random_string() for _ in range(3)]

    example_dict = {
        keys[0]: torch.randn(2, 3),
        keys[1]: torch.randint(0, 10, (4,)),
        keys[2]: torch.randn(1, 5, 2),
    }

    example_tensordict = TensorDict(example_dict)

    print(f"Original dict keys: {list(example_dict.keys())}")
    print(f"TensorDict keys:    {list(example_tensordict.keys())}")
    print(
        f"Order preserved: {list(example_dict.keys()) == list(example_tensordict.keys())}"
    )
