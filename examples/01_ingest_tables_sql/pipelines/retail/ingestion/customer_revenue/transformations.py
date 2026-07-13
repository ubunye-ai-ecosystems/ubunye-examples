"""Aggregate bakehouse sales into revenue by franchise and by product.

The business rule, and only the business rule. Where the data comes from and
where it goes is config.yaml's problem; this file never names a table.
"""

from __future__ import annotations

from typing import Any, Dict

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from ubunye.core.interfaces import Task


class CustomerRevenue(Task):
    """Revenue per franchise per month, and per product per month."""

    def transform(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        transactions = sources["transactions"]
        franchises = sources["franchises"]

        monthly = transactions.withColumn(
            "revenue_month", F.date_format(F.col("dateTime"), "yyyy-MM")
        )

        return {
            "franchise_revenue": self._by_franchise(monthly, franchises),
            "product_revenue": self._by_product(monthly),
        }

    def _by_franchise(self, monthly: DataFrame, franchises: DataFrame) -> DataFrame:
        """Revenue per franchise, with the franchise's name and city attached.

        The franchise table is tiny and the transaction table is not, so this is
        a broadcast join — Spark ships the small side to every executor instead
        of shuffling the large one across the network.
        """
        return (
            monthly.groupBy("franchiseID", "revenue_month")
            .agg(
                F.sum("totalPrice").alias("revenue"),
                F.sum("quantity").alias("items_sold"),
                F.countDistinct("transactionID").alias("transactions"),
                F.countDistinct("customerID").alias("customers"),
            )
            .join(
                F.broadcast(
                    franchises.select(
                        F.col("franchiseID"),
                        F.col("name").alias("franchise_name"),
                        F.col("city").alias("franchise_city"),
                        F.col("country").alias("franchise_country"),
                    )
                ),
                on="franchiseID",
                how="left",
            )
            .withColumn(
                "avg_transaction_value",
                F.round(F.col("revenue") / F.col("transactions"), 2),
            )
            .withColumn("ingested_at", F.current_timestamp())
        )

    def _by_product(self, monthly: DataFrame) -> DataFrame:
        """Revenue per product per month, ranked within each month."""
        from pyspark.sql import Window

        by_month = Window.partitionBy("revenue_month").orderBy(F.col("revenue").desc())

        return (
            monthly.groupBy("product", "revenue_month")
            .agg(
                F.sum("totalPrice").alias("revenue"),
                F.sum("quantity").alias("units"),
                F.countDistinct("transactionID").alias("transactions"),
            )
            .withColumn("revenue_rank", F.rank().over(by_month))
            .withColumn("ingested_at", F.current_timestamp())
        )
