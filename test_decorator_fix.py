"""
Quick test to verify the decorator fix
"""
import sys
sys.path.insert(0, ".")

# Test the decorator
from multi_src.helpers import for_each_geo_with_data_filter

# Create a simple test function
def test_func(contributions, mmm, data, use_log_scale=True, **kwargs):
    """Test function that mimics plot_all_contributions signature"""
    print(f"Called with:")
    print(f"  contributions: {type(contributions)}")
    print(f"  mmm: {mmm}")
    print(f"  data: {type(data)}")
    print(f"  use_log_scale: {use_log_scale}")
    print(f"  kwargs: {kwargs}")
    return "success"

# Test 1: Non-multidimensional contributions (should pass through)
print("=" * 70)
print("TEST 1: Non-multidimensional contributions")
print("=" * 70)

contributions_simple = {
    'channels': {},
}

try:
    decorated_func = for_each_geo_with_data_filter(test_func)
    result = decorated_func(
        contributions=contributions_simple,
        mmm="test_mmm",
        data="test_data",
        use_log_scale=False,
        custom_arg="custom_value"
    )
    print(f"✓ Result: {result}\n")
except Exception as e:
    print(f"✗ Error: {e}\n")

# Test 2: Check that it requires data as keyword argument
print("=" * 70)
print("TEST 2: Verify data must be keyword argument")
print("=" * 70)

try:
    # This should fail because data is not a keyword argument
    decorated_func = for_each_geo_with_data_filter(test_func)
    result = decorated_func(contributions_simple, "test_mmm", "test_data")
    print(f"✗ Should have raised ValueError, but got: {result}\n")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {e}\n")
except Exception as e:
    print(f"✗ Wrong error type: {e}\n")

print("=" * 70)
print("TESTS COMPLETED")
print("=" * 70)
