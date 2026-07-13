"""Seed the source tables off Databricks — same schema, synthetic data.

The examples read `samples.nyctaxi` and `samples.bakehouse`, which exist only inside a
Databricks workspace. This script creates tables with **exactly the same names, columns
and types** in a local Hive metastore, filled with data we generate ourselves.

**The data is not the point. The process is.**

The pipelines under test do not change by a single character. They still read
`<catalog>.bakehouse.sales_transactions`, still push a join down as SQL, still train a
model, still enforce a contract, still gate and register. What changes is the catalog
they resolve against — `samples` on Databricks, `spark_catalog` here — and that is one
environment variable.

So the numbers will differ between platforms, and they are *supposed to*. A model
trained on different rows scores differently; that is not a portability failure, it is
arithmetic. What must be identical is the **shape**: the same tasks run, the same
columns come out, the same gates fire, the same contract catches the same *kinds* of
breach. Example 11 is the one that asserts byte-identical output, because it is the one
whose input is byte-identical.

Determinism still matters, though — same seed, same rows, every run — or a failing test
here would be impossible to reproduce.

Usage:
    python platforms/seed.py            # writes into spark_catalog
"""

from __future__ import annotations

import datetime as dt
import os
import random
from typing import Any, List, Tuple

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Same seed, same rows, on every platform and every run. A flaky fixture makes a
# failing pipeline impossible to tell apart from a failing generator.
SEED = 42

CATALOG = os.environ.get("SOURCE_CATALOG", "spark_catalog")
N_TRIPS = 20_000
N_TRANSACTIONS = 5_000
N_CUSTOMERS = 400
N_FRANCHISES = 12
N_REVIEWS = 200

PRODUCTS = [
    "Golden Gate Ginger",
    "Outback Oatmeal",
    "Austin Almond Biscotti",
    "Tokyo Tidbits",
    "Orchard Oasis",
    "Pearly Pies",
]
PAYMENTS = ["visa", "mastercard", "amex", "discover"]
CITIES = [
    ("Johannesburg", "ZA"),
    ("Cape Town", "ZA"),
    ("Lagos", "NG"),
    ("Nairobi", "KE"),
    ("Accra", "GH"),
    ("Cairo", "EG"),
]


def _taxi(rng: random.Random) -> List[Tuple[Any, ...]]:
    """NYC-taxi-shaped trips.

    The fare is a real function of distance and duration plus noise — not random. A
    model has to have something to learn, or example 04's quality gate would fail and
    the pipeline would (correctly) refuse to register anything, and we would have
    proved nothing about portability.
    """
    base = dt.datetime(2026, 2, 1)
    rows = []
    for _ in range(N_TRIPS):
        pickup = base + dt.timedelta(minutes=rng.randint(0, 40 * 24 * 60))
        distance = round(abs(rng.lognormvariate(0.6, 0.7)) + 0.3, 2)
        minutes = max(1.0, distance * rng.uniform(2.5, 4.5) + rng.uniform(0, 4))
        fare = round(2.5 + 2.6 * distance + 0.35 * minutes + rng.gauss(0, 0.8), 2)
        rows.append(
            (
                pickup,
                pickup + dt.timedelta(minutes=minutes),
                distance,
                max(fare, 3.0),
                rng.randint(10001, 11250),
                rng.randint(10001, 11250),
            )
        )
    return rows


def _transactions(rng: random.Random) -> List[Tuple[Any, ...]]:
    """Bakehouse-shaped sales.

    Prices are integers, exactly as the real table has them (`unitPrice: bigint`), and
    totalPrice reconciles to quantity x unitPrice — because example 07's data contract
    checks precisely that, and a seed that quietly broke the rule would make the
    contract look like it was failing when it was working.
    """
    base = dt.datetime(2026, 1, 1)
    rows = []
    for i in range(N_TRANSACTIONS):
        quantity = rng.randint(1, 6)
        unit_price = rng.randint(3, 12)
        rows.append(
            (
                900_000_000 + i,
                rng.randint(1, N_CUSTOMERS),
                rng.randint(1, N_FRANCHISES),
                base + dt.timedelta(minutes=rng.randint(0, 150 * 24 * 60)),
                rng.choice(PRODUCTS),
                quantity,
                unit_price,
                quantity * unit_price,  # must reconcile — see the contract in 07
                rng.choice(PAYMENTS),
                rng.randint(4_000_000_000_000_000, 4_999_999_999_999_999),
            )
        )
    return rows


def main() -> int:
    rng = random.Random(SEED)

    spark = SparkSession.builder.appName("ubunye:seed").enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    spark.sql("CREATE DATABASE IF NOT EXISTS nyctaxi")
    spark.sql("CREATE DATABASE IF NOT EXISTS bakehouse")

    tables = {
        "nyctaxi.trips": (
            _taxi(rng),
            "tpep_pickup_datetime timestamp, tpep_dropoff_datetime timestamp, "
            "trip_distance double, fare_amount double, pickup_zip int, dropoff_zip int",
        ),
        "bakehouse.sales_transactions": (
            _transactions(rng),
            "transactionID bigint, customerID bigint, franchiseID bigint, dateTime timestamp, "
            "product string, quantity bigint, unitPrice bigint, totalPrice bigint, "
            "paymentMethod string, cardNumber bigint",
        ),
        "bakehouse.sales_franchises": (
            [
                (
                    i,
                    f"Bakehouse {CITIES[i % len(CITIES)][0]} {i}",
                    CITIES[i % len(CITIES)][0],
                    f"District {i % 4}",
                    str(2000 + i),
                    CITIES[i % len(CITIES)][1],
                    rng.choice(["small", "medium", "large"]),
                    round(rng.uniform(-30, 35), 4),
                    round(rng.uniform(-35, 32), 4),
                    rng.randint(1, 5),
                )
                for i in range(1, N_FRANCHISES + 1)
            ],
            "franchiseID bigint, name string, city string, district string, zipcode string, "
            "country string, size string, longitude double, latitude double, supplierID bigint",
        ),
        "bakehouse.sales_customers": (
            [
                (
                    i,
                    f"First{i}",
                    f"Last{i}",
                    f"customer{i}@example.com",
                    f"+27-11-555-{i:04d}",
                    f"{i} Main Road",
                    CITIES[i % len(CITIES)][0],
                    "Gauteng",
                    CITIES[i % len(CITIES)][1],
                    "Africa",
                    2000 + i,
                    rng.choice(["female", "male", "other"]),
                )
                for i in range(1, N_CUSTOMERS + 1)
            ],
            "customerID bigint, first_name string, last_name string, email_address string, "
            "phone_number string, address string, city string, state string, country string, "
            "continent string, postal_zip_code bigint, gender string",
        ),
        "bakehouse.media_customer_reviews": (
            [
                (
                    # Long enough to actually chunk, and opinionated enough for a
                    # sentiment model to have something to learn. Example 06's teacher
                    # LLM does not run off Databricks, but 03's chunker does.
                    (
                        "The pastries here are consistently excellent and the staff remember "
                        "your order. I have been coming every week for a year and it has never "
                        "once disappointed me. "
                        if i % 3
                        else "Waited forty minutes for a cold coffee and a stale croissant. "
                        "The queue was out of the door and nobody seemed to care. I will not "
                        "be returning to this branch. "
                    )
                    * 3,
                    rng.randint(1, N_FRANCHISES),
                    dt.datetime(2026, 1, 1) + dt.timedelta(days=rng.randint(0, 150)),
                    i,
                )
                for i in range(1, N_REVIEWS + 1)
            ],
            "review string, franchiseID bigint, review_date timestamp, new_id int",
        ),
    }

    for name, (rows, schema) in tables.items():
        df = spark.createDataFrame(rows, schema)
        (
            df.write.format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(f"{CATALOG}.{name}")
        )
        print(f"  {CATALOG}.{name:<34} {df.count():>6} rows")

    # A pipeline that reads an empty table succeeds and produces nothing, and that is
    # the single most convincing kind of green build there is.
    for name in tables:
        n = spark.table(f"{CATALOG}.{name}").count()
        assert n > 0, f"{name} seeded empty"

    print()
    print(f"seeded into catalog '{CATALOG}' — same schema as samples.*, different rows")
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
