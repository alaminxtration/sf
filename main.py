
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, to_timestamp, year, month, weekofyear, avg, min, max, desc, count
from pyspark.sql.types import *
import os

#  Set Hadoop home for Windows compatibility
os.environ['HADOOP_HOME'] = 'C:\\hadoop-2.8.3'  # Update with your Hadoop path

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("FireCallAnalysis") \
    .config("spark.sql.debug.maxToStringFields", "100") \
    .getOrCreate()

# Step 1: Define Schema
fire_schema = StructType([
    StructField('CallNumber', IntegerType(), True),
    StructField('UnitID', StringType(), True),
    StructField('IncidentNumber', IntegerType(), True),
    StructField('CallType', StringType(), True),
    StructField('CallDate', StringType(), True),
    StructField('WatchDate', StringType(), True),
    StructField('CallFinalDisposition', StringType(), True),
    StructField('AvailableDtTm', StringType(), True),
    StructField('Address', StringType(), True),
    StructField('City', StringType(), True),
    StructField('Zipcode', IntegerType(), True),
    StructField('Battalion', StringType(), True),
    StructField('StationArea', StringType(), True),
    StructField('Box', StringType(), True),
    StructField('OriginalPriority', StringType(), True),
    StructField('Priority', StringType(), True),
    StructField('FinalPriority', IntegerType(), True),
    StructField('ALSUnit', BooleanType(), True),
    StructField('CallTypeGroup', StringType(), True),
    StructField('NumAlarms', IntegerType(), True),
    StructField('UnitType', StringType(), True),
    StructField('UnitSequenceInCallDispatch', IntegerType(), True),
    StructField('FirePreventionDistrict', StringType(), True),
    StructField('SupervisorDistrict', StringType(), True),
    StructField('Neighborhood', StringType(), True),
    StructField('Location', StringType(), True),
    StructField('RowID', StringType(), True),
    StructField('Delay', FloatType(), True)
])

# Step 2: Load Data
file_path = "C:/Users/Alamin/PycharmProjects/bootstrappython/pythonProject/sf-fire-calls.txt"
fire_df = spark.read.csv(file_path, schema=fire_schema, header=True)

#  Maintain consistent DataFrame reference through transformations
new_fire_df = fire_df.withColumnRenamed("Delay", "ResponseDelayedinMins")
# Task 1: Select specific columns and filter "Medical Incident"
few_fire_df = fire_df.select("IncidentNumber", "AvailableDtTm", "CallType") \
    .where(col("CallType") != "Medical Incident")
print("\nTask 1: Filtered calls (excluding Medical Incident):")
few_fire_df.show(5, truncate=False)

# Task 2: Count distinct CallTypes
print("\nTask 2: Distinct call type count:")
fire_df.select("CallType").where(col("CallType").isNotNull()) \
    .agg(countDistinct("CallType").alias("DistinctCallTypes")).show()

# Task 3: List distinct CallTypes
print("\nTask 3: Distinct call types:")
fire_df.select("CallType").where(col("CallType").isNotNull()).distinct().show(10, False)

# Task 4: Rename "Delay" to "ResponseDelayedinMins" and filter >5 mins
print("\nTask 4: Delays > 5 minutes:")
new_fire_df = fire_df.withColumnRenamed("Delay", "ResponseDelayedinMins")
new_fire_df.select("ResponseDelayedinMins").where(col("ResponseDelayedinMins") > 5).show(5, False)
# Task 5: Convert timestamps correctly
cleaned_df = new_fire_df \
    .withColumn('IncidentDate', to_timestamp(col('CallDate'), 'MM/dd/yyyy')) \
    .drop('CallDate') \
    .withColumn("OnWatchDate", to_timestamp(col("WatchDate"), "MM/dd/yyyy")) \
    .drop("WatchDate") \
    .withColumn("AvailableDtTS", to_timestamp(col("AvailableDtTm"), "MM/dd/yyyy hh:mm:ss a")) \
    .drop("AvailableDtTm")
print("\nTask 5: Converted timestamps:")
cleaned_df.select("IncidentDate", "OnWatchDate", "AvailableDtTS").show(5, True)#1111
# Task 6: Find most common CallTypes
print("\nTask 6: Most frequent call types:")
fire_df.groupBy("CallType").count().orderBy(desc("count")).show(10, False)

# Updated Task 7 using new_fire_df
print("\nTask 7: Response time statistics:")
new_fire_df.select(
    avg("ResponseDelayedinMins").alias("AverageResponseTime"),
    min("ResponseDelayedinMins").alias("MinResponseTime"),
    max("ResponseDelayedinMins").alias("MaxResponseTime")
).show()
# Prepare 2018 data for Tasks 8-12
fire_2018_df = cleaned_df.filter(year("IncidentDate") == 2018) #111
# Task 8: Find all fire call types in 2018
print("\nTask 8: 2018 call types:")
fire_2018_df = cleaned_df.filter(year("IncidentDate") == 2018)
fire_2018_df.select("CallType").distinct().show()

# Task 9: Find the month in 2018 with the highest number of fire calls
print("\nTask 9: Busiest month in 2018:")
fire_2018_df.groupBy(month("IncidentDate").alias("Month")).count().orderBy(desc("count")).show(1)

# Task 10: Find the neighborhood with the most fire calls in 2018
print("\nTask 10: Top neighborhood in 2018:")
fire_2018_df.groupBy("Neighborhood").count().orderBy(desc("count")).show(1, False)

# Task 11: Find neighborhoods with the worst response times in 2018
print("\nTask 11: Worst response neighborhoods 2018:")
fire_2018_df.groupBy("Neighborhood") \
    .agg(avg("ResponseDelayedinMins").alias("AvgResponseTime")) \
    .orderBy(desc("AvgResponseTime")) \
    .show(5, False)

# Task 12: Find the week in 2018 with the most fire calls
print("\nTask 12: Busiest week in 2018:")
fire_2018_df.groupBy(weekofyear("IncidentDate").alias("Week")) \
    .count().orderBy(desc("count")).show(1)

# Replace invalid correlation analysis with proper frequency count
# Task 13 (Corrected): Analyze fire calls per Zipcode
print("\nTop 10 Zipcodes by Fire Call Frequency:")
fire_calls_by_zip = cleaned_df.groupBy("Zipcode") \
    .agg(count("*").alias("TotalCalls")) \
    .filter(col("Zipcode").isNotNull()) \
    .orderBy(desc("TotalCalls")) \
    .limit(10)

fire_calls_by_zip.show(truncate=False)

# Task 14: Parquet operations (Fixed indentation)
print("\nWriting Parquet file...")
cleaned_df.write.mode("overwrite").parquet("fire_calls.parquet")

print("\nReading from Parquet:")
parquet_df = spark.read.parquet("fire_calls.parquet")
parquet_df.show(5, vertical=True)  # Better for wide datasets

# SQL Operations
cleaned_df.createOrReplaceTempView("fire_calls")

print("\nStructure Fire Incidents:")
spark.sql("""
    SELECT IncidentNumber, IncidentDate, Neighborhood, ResponseDelayedinMins 
    FROM fire_calls 
    WHERE CallType = 'Structure Fire'
    ORDER BY IncidentDate DESC
    LIMIT 5
""").show(truncate=False)

# Stop Spark Session
spark.stop()
