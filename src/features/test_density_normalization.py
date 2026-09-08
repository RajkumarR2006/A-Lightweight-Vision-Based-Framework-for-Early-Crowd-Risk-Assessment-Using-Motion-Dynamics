from density import normalize_density


test_values = [
    0.0025,
    0.0100,
    0.0250,
    0.0400,
    0.0500,
    0.0617,
    0.1000,
    1.0000
]


for density in test_values:

    normalized = normalize_density(density)

    print(
        f"Density: {density:.4f} "
        f"-> Normalized: {normalized:.4f}"
    )