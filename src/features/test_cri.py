from cri import compute_cri


# --------------------------------------------------
# Test 1: Stable low CPI
# --------------------------------------------------

cpi_values = [
    0.10,
    0.12,
    0.11,
    0.13,
    0.12
]

cri = compute_cri(cpi_values, window_size=5)

print("=" * 60)
print("Test 1 - Stable Low CPI")
print(f"CPI values : {cpi_values}")
print(f"CRI        : {cri:.4f}")


# --------------------------------------------------
# Test 2: Increasing CPI
# --------------------------------------------------

cpi_values = [
    0.30,
    0.45,
    0.60,
    0.75,
    0.85
]

cri = compute_cri(cpi_values, window_size=5)

print("=" * 60)
print("Test 2 - Increasing CPI")
print(f"CPI values : {cpi_values}")
print(f"CRI        : {cri:.4f}")


# --------------------------------------------------
# Test 3: One sudden spike
# --------------------------------------------------

cpi_values = [
    0.20,
    0.20,
    0.20,
    0.20,
    0.90
]

cri = compute_cri(cpi_values, window_size=5)

print("=" * 60)
print("Test 3 - Single Spike")
print(f"CPI values : {cpi_values}")
print(f"CRI        : {cri:.4f}")


# --------------------------------------------------
# Test 4: Persistent high CPI
# --------------------------------------------------

cpi_values = [
    0.80,
    0.82,
    0.85,
    0.88,
    0.90
]

cri = compute_cri(cpi_values, window_size=5)

print("=" * 60)
print("Test 4 - Persistent High CPI")
print(f"CPI values : {cpi_values}")
print(f"CRI        : {cri:.4f}")

print("=" * 60)