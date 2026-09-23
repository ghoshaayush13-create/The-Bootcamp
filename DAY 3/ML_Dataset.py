import numpy as np
import pandas as pd

# Random seed
np.random.seed(15)

# Number of students
n = 500

# Generate data
student_id = np.arange(1001, 1001 + n)

age = np.random.randint(18, 23, n)
attendance = np.random.uniform(55, 100, n)
study_hours = np.random.uniform(1, 8, n)
assignment_score = np.random.uniform(40, 100, n)
midterm_score = np.random.uniform(35, 100, n)
projects_completed = np.random.randint(0, 8, n)
coding_hours = np.random.uniform(0, 6, n)
communication_score = np.random.uniform(40, 100, n)
internship = np.random.randint(0, 2, n)
backlogs = np.random.randint(0, 5, n)
cgpa = np.random.uniform(5.0, 10.0, n)

# Calculate placement score
placement_score = (
    0.25 * cgpa +
    0.15 * attendance / 10 +
    0.15 * assignment_score / 10 +
    0.15 * midterm_score / 10 +
    0.10 * projects_completed +
    0.10 * coding_hours +
    0.10 * internship -
    0.20 * backlogs
)

# Target variable
placement = (placement_score > 3.8).astype(int)

# Create DataFrame
data = pd.DataFrame({
    "Student_ID": student_id,
    "Age": age,
    "Attendance": attendance,
    "Study_Hours": study_hours,
    "Assignment_Score": assignment_score,
    "Midterm_Score": midterm_score,
    "Projects_Completed": projects_completed,
    "Coding_Hours": coding_hours,
    "Communication_Score": communication_score,
    "Internship": internship,
    "Backlogs": backlogs,
    "CGPA": cgpa,
    "Placement": placement
})

# Round decimal values
data = data.round(2)

# Save CSV
data.to_csv("student_placement_dataset.csv", index=False)

# Display information
print("Dataset created successfully!")
print("\nFirst 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nDataset information:")
print(data.info())

print("\nPlacement distribution:")
print(data["Placement"].value_counts())
