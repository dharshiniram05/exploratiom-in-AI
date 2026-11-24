

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = {
    "Student_ID": [
        "S101", "S102", "S103", "S104", "S105",
        "S106", "S107", "S108", "S109", "S110",
        "S111", "S112", "S113", "S114", "S115",
        "S116", "S117", "S118", "S119", "S120",
        "S121", "S101"
    ],

    "Name": [
        "Aarav", "Diya", "Kavin", "Meera", "Arjun",
        "Nila", "Rohan", "Ishita", "Vikram", "Ananya",
        "Rahul", "Priya", "Karthik", "Sneha", "Varun",
        "Asha", "Ravi", "Divya", "Manoj", "Keerthi",
        "Sanjay", "Aarav"
    ],

    "Department": [
        "CSE", "ECE", "CSE", "IT", "ECE",
        "CSE", "IT", "CSE", "ECE", "IT",
        "CSE", None, "ECE", "IT", "CSE",
        "ECE", "CSE", "IT", "ECE", "CSE",
        "IT", "CSE"
    ],

    "Attendance_Percentage": [
        92, 85, 78, 96, np.nan,
        88, 74, 91, 67, 83,
        np.nan, 89, 95, 80, 87,
        np.nan, 76, 93, 84, 90,
        82, 92
    ],

    "Assignment_Score": [
        88, 76, 81, 93, 72,
        81, 69, 90, 73, 85,
        78, 92, np.nan, 88, 75,
        84, 79, 95, 70, 86,
        91, 88
    ],

    "Exam_Score": [
        91, 82, 79, 95, 75,
        86, 68, 92, 71, 88,
        80, 94, 85, np.nan, 77,
        89, 73, 96, 81, 90,
        87, 91
    ]
}

# Create and save CSV file
sample_df = pd.DataFrame(data)

sample_df.to_csv(
    "experiment_8_student_performance.csv",
    index=False
)

# Read CSV file
df = pd.read_csv("experiment_8_student_performance.csv")

print("----- ORIGINAL DATASET -----")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

# Handle missing numerical values
num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Handle missing categorical values
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Calculate average score
df["Average_Score"] = df[
    ["Assignment_Score", "Exam_Score"]
].mean(axis=1)

# Classify student performance
df["Performance"] = pd.cut(
    df["Average_Score"],
    bins=[-1, 50, 70, 85, 100],
    labels=[
        "Needs Improvement",
        "Average",
        "Good",
        "Excellent"
    ]
)

print("\n----- CLEANED DATASET -----")
print(df.head(10))

print("\nCleaned Dataset Shape:", df.shape)

# Statistical analysis
print("\n----- STATISTICAL SUMMARY -----")
print(df.describe().round(2))

# Department-wise analysis
dept_analysis = df.groupby(
    "Department"
)["Exam_Score"].mean()

print("\n----- DEPARTMENT-WISE AVERAGE -----")
print(dept_analysis.round(2))

# Performance distribution
print("\n----- PERFORMANCE DISTRIBUTION -----")
print(df["Performance"].value_counts())

# Visualization 1
plt.figure(figsize=(8, 5))

dept_analysis.plot(
    kind="bar",
    color="skyblue",
    edgecolor="black"
)

plt.title("Department-wise Average Exam Score")
plt.xlabel("Department")
plt.ylabel("Average Exam Score")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Visualization 2
plt.figure(figsize=(8, 5))

df["Performance"].value_counts().plot(
    kind="bar",
    color="orange",
    edgecolor="black"
)

plt.title("Student Performance Distribution")
plt.xlabel("Performance Category")
plt.ylabel("Number of Students")
plt.xticks(rotation=15)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

print("\nData Cleaning and EDA Completed Successfully!")