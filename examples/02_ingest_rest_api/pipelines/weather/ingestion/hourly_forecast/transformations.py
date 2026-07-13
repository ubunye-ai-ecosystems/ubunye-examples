"""Fan one JSON document out into one row per hour.

A REST payload is almost never shaped like a table. Open-Meteo returns a single
document in which `hourly.time` is an array of 168 timestamps and
`hourly.temperature_2m` is a parallel array of 168 temperatures — the reader hands
that to you as ONE row containing several long arrays.

Turning that back into rows is the business rule. `arrays_zip` stitches the
parallel arrays together element-wise (so hour i keeps its own temperature) and
`explode` turns each element into a row. Zipping first is the point: exploding the
arrays separately would give you a cross product of 168 × 168 rows, most of them
pairing a timestamp with somebody else's temperature.
"""

from __future__ import annotations

from typing import Any, Dict

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from ubunye.core.interfaces import Task

CITY = "Johannesburg"

MEASURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
]


class HourlyForecast(Task):
    """One Open-Meteo document -> one row per forecast hour."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        return {"hourly_forecast": self._explode(sources["forecast"])}

    def _explode(self, forecast: DataFrame) -> DataFrame:
        zipped = F.arrays_zip(
            F.col("hourly.time").alias("time"),
            *[F.col(f"hourly.{m}").alias(m) for m in MEASURES],
        )

        return (
            forecast.select(
                F.col("latitude"),
                F.col("longitude"),
                F.col("timezone"),
                F.explode(zipped).alias("hour"),
            )
            .select(
                F.lit(CITY).alias("city"),
                F.col("latitude").cast("double").alias("latitude"),
                F.col("longitude").cast("double").alias("longitude"),
                F.col("timezone"),
                # Open-Meteo returns local ISO timestamps like "2026-07-13T14:00".
                F.to_timestamp(F.col("hour.time")).alias("observed_at"),
                *[F.col(f"hour.{m}").cast("double").alias(m) for m in MEASURES],
            )
            .withColumn("forecast_date", F.to_date("observed_at"))
            .withColumn("hour_of_day", F.hour("observed_at"))
            .withColumn(
                "conditions",
                F.when(F.col("precipitation") > 5, "heavy rain")
                .when(F.col("precipitation") > 0.2, "rain")
                .when(F.col("wind_speed_10m") > 40, "windy")
                .when(F.col("temperature_2m") > 30, "hot")
                .when(F.col("temperature_2m") < 5, "cold")
                .otherwise("fair"),
            )
            .withColumn("ingested_at", F.current_timestamp())
            .filter(F.col("observed_at").isNotNull())
        )
