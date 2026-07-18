"""Bearing vibration feature transformations (Spark).

Computes rolling vibration statistics per bearing, point-in-time correct:
every window ends at the current row's timestamp — no future data.
"""

from pyspark.ml import Transformer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class BearingVibrationTransformer(Transformer):
    def _transform(self, df: DataFrame) -> DataFrame:
        # 1-hour rolling window per bearing, ending at current row (inclusive).
        # rangeBetween is in SECONDS relative to the current row's timestamp.
        one_hour = (
            Window.partitionBy("device_id")
            .orderBy(F.col("timestamp").cast("long"))
            .rangeBetween(-3600, 0)
        )

        # Expanding window: everything from the bearing's first reading to now —
        # used as the bearing's own historical baseline.
        baseline = (
            Window.partitionBy("device_id")
            .orderBy(F.col("timestamp").cast("long"))
            .rangeBetween(Window.unboundedPreceding, 0)
        )

        df = df.withColumn("rms_value", F.col("value"))
        df = df.withColumn("rolling_mean_1h", F.mean("value").over(one_hour))
        df = df.withColumn("rolling_std_1h", F.stddev("value").over(one_hour))
        df = df.withColumn(
            "deviation_from_baseline", F.col("value") - F.mean("value").over(baseline)
        )

        return df.select(
            "device_id",
            "timestamp",
            "rms_value",
            "rolling_mean_1h",
            "rolling_std_1h",
            "deviation_from_baseline",
        )
