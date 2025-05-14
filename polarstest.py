import pandas as pd
import polars as pl
import time

# PANDAS
start_time = time.time()
df_pd = pd.read_csv("data.csv")
end_time = time.time()
print(f"Pandas read_csv took {end_time - start_time:.6f} seconds")

# POLARS
start_time = time.time()
df_pl = pl.read_csv("data.csv")
end_time = time.time()
print(f"Polars read_csv took {end_time - start_time:.6f} seconds")


# PANDAS
start_time = time.time()
df_pd_italy = pd.read_csv("data.csv")
df_pd_italy["date"] = pd.to_datetime(df_pd_italy["date"])  # Ensure 'date' is in datetime format
df_pd_italy = df_pd_italy[df_pd_italy["location"] == "Italy"]
df_pd_italy = df_pd_italy[["date", "sales"]]
end_time = time.time()
print(f"Pandas filtering and selection took {end_time - start_time:.6f} seconds")


# POLARS
start_time = time.time()
df_pl_italy = (
    pl.read_csv("data.csv")
      .with_columns(pl.col("date").str.to_date("%Y-%m-%d %H:%M:%S"))
      .filter(pl.col("location") == "Italy")
      .select(pl.col(["date", "sales"]))
)
end_time = time.time()
print(f"Polars filtering and selection took {end_time - start_time:.6f} seconds")

# PANDAS
start_time = time.time()
df_pd["date"] = pd.to_datetime(df_pd["date"])  # Ensure 'date' is in datetime format
df_pd_res = (
    df_pd
        .groupby([df_pd["date"].dt.year, df_pd["location"]])  # Use .dt.year after conversion
        .agg(
            total_sales=("sales", "sum"),
            avg_sales=("sales", "mean")
        )
        .reset_index()
        .sort_values(by=["date", "total_sales"], ascending=[True, False])
)
end_time = time.time()
print(f"Pandas groupby and aggregation took {end_time - start_time:.6f} seconds")


# POLARS
start_time = time.time()
df_pl_res = (
    pl.read_csv("data.csv")
        .with_columns(pl.col("date").str.to_date("%Y-%m-%d %H:%M:%S"))
        .group_by([pl.col("date").dt.year(), pl.col("location")])
        .agg(
            pl.sum("sales").alias("total_sales"),
            pl.mean("sales").alias("avg_sales")
        )
        .sort(by=["date", "total_sales"], descending=[False, True])
)
end_time = time.time()
print(f"Polars groupby and aggregation took {end_time - start_time:.6f} seconds")


# POLARS WITH LAZY EVALUATION
start_time = time.time()
df_pl_lazy_query = (
    pl.scan_csv("data.csv")
        .with_columns(pl.col("date").str.to_date("%Y-%m-%d %H:%M:%S"))
        .group_by([pl.col("date").dt.year(), pl.col("location")])
        .agg(
            pl.sum("sales").alias("total_sales"),
            pl.mean("sales").alias("avg_sales")
        )
        .sort(by=["date", "total_sales"], descending=[False, True])
)
df_pl_lazy_res = df_pl_lazy_query.collect()
end_time = time.time()
print(f"Polars lazy evaluation took {end_time - start_time:.6f} seconds")