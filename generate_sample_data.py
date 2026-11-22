"""TEST-ONLY synthetic data so you can check the pipeline runs end to end.
DO NOT submit results from this file - replace with REAL Sri Lankan data."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 2500
makes = {"Toyota": ["Aqua", "Prius", "Vitz", "Corolla", "Axio"], "Honda": ["Fit", "Vezel", "Civic", "Grace"],
         "Suzuki": ["Alto", "Wagon R", "Swift"], "Nissan": ["Leaf", "March", "X-Trail"], "Mitsubishi": ["Montero", "Lancer"]}
base = {"Toyota": 1.25, "Honda": 1.15, "Suzuki": 0.85, "Nissan": 1.0, "Mitsubishi": 1.2}
rows = []
for _ in range(n):
    mk = rng.choice(list(makes)); md = rng.choice(makes[mk])
    year = int(rng.integers(2005, 2025)); age = 2026 - year
    km = max(2000, int(rng.normal(9000 * age + 10000, 15000)))
    cc = int(rng.choice([660, 1000, 1300, 1500, 1800, 2000, 2400]))
    fuel = rng.choice(["Petrol", "Hybrid", "Diesel", "Electric"], p=[.55, .3, .12, .03])
    trans = rng.choice(["Automatic", "Manual"], p=[.8, .2])
    dist = rng.choice(["Colombo", "Gampaha", "Kandy", "Galle", "Kurunegala", "Jaffna"])
    price = 6e6 * base[mk] * (cc / 1300) ** .6 * np.exp(-.07 * age) * np.exp(-km / 4e5)
    price *= {"Hybrid": 1.25, "Electric": 1.3, "Diesel": 1.1, "Petrol": 1.0}[fuel] * (1 if trans == "Automatic" else .9)
    price *= {"Colombo": 1.05}.get(dist, 1.0) * rng.lognormal(0, .12)
    rows.append([mk, md, year, f"{km:,} km", f"{cc} cc", fuel, trans, dist, f"Rs. {int(price):,}"])
pd.DataFrame(rows, columns=["make", "model", "year", "mileage", "engine", "fuel", "transmission", "district", "price"]
             ).to_csv("data/vehicles_SAMPLE_synthetic.csv", index=False)
print("Wrote data/vehicles_SAMPLE_synthetic.csv (synthetic - for testing only)")
