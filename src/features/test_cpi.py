from cpi import compute_cpi


# --------------------------------------------------
# Normalization parameters
# --------------------------------------------------

DENSITY_MIN = 0.0025
DENSITY_MAX = 0.0617

MII_MIN = 0.2298
MII_MAX = 0.6886


# --------------------------------------------------
# Test Cases
# --------------------------------------------------

test_cases = [
    {
        "name": "Low congestion",
        "density": 0.0100,
        "mii": 0.2500,
        "dci": 0.5500
    },
    {
        "name": "Moderate congestion",
        "density": 0.0400,
        "mii": 0.4500,
        "dci": 0.7500
    },
    {
        "name": "High congestion",
        "density": 0.0600,
        "mii": 0.6500,
        "dci": 0.9500
    }
]


# --------------------------------------------------
# Run Tests
# --------------------------------------------------

for case in test_cases:

    cpi = compute_cpi(
        density=case["density"],
        mii=case["mii"],
        dci=case["dci"],
        density_min=DENSITY_MIN,
        density_max=DENSITY_MAX,
        mii_min=MII_MIN,
        mii_max=MII_MAX
    )

    print("=" * 60)
    print(f"Case    : {case['name']}")
    print(f"Density : {case['density']:.4f}")
    print(f"MII     : {case['mii']:.4f}")
    print(f"DCI     : {case['dci']:.4f}")
    print(f"CPI     : {cpi:.4f}")