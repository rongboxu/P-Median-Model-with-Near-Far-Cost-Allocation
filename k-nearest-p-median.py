# %%
import pandas as pd
import numpy as np
from pulp import *

# %% [markdown]
# ### the p-median model with k nearest facilities

# %% [markdown]
# For this first stage, only k nearest facilities will be imported in the model.
# 1. After importing original data, I create the new table only containing k nearest facilities.
# 2. To test the method, I use the test IOE data as I used for capacitated p-median case.
# So in this model, the capacity constraint is still live.
# 3. In a broader case, it should be that, the sum of demand value of the facility can serve is no more than its capacity.

# %%
# import data
time = pd.read_csv("data/example_subject_student_school_journeys.csv")
time_table = (
    time.pivot_table(
        columns="school", fill_value=10000, index="student", sort=False, values="time",
    )
    .astype(int)
    .values
)
students_df = pd.read_csv("data/example_subject_students.csv")
schools_df = pd.read_csv("data/example_subject_schools.csv")

# %%
# find the k nearest facility from distance matrix

# Define the value of k
k = 5

# Create an empty list to hold the rows
rows = []

# Iterate over each client point and add the k nearest facility indices to the new pivot table
for client_idx in range(time_table.shape[0]):
    distances = time_table[client_idx]
    nearest_index = np.argsort(distances)[:k]
    # Append rows to the list
    for i in nearest_index:
        row = {"client_id": client_idx, "facility_id": i, "distance": distances[i]}
        rows.append(row)

# Create a new_distance_df dataframe by concatenating the rows
new_distance_df = pd.DataFrame(rows, columns=["client_id", "facility_id", "distance"])

# %%
new_distance_df.head(10)

# %%
# transform the new df to the pivot table
new_time_table = (
    new_distance_df.pivot_table(
        columns="facility_id",
        fill_value=10000,
        index="client_id",
        sort=False,
        values="distance",
    )
    .astype(int)
    .values
)

# %%
new_time_table[:4]

# %%
# create capacity df
new_school_index = new_distance_df["facility_id"].sort_values().unique()
new_school_df = pd.DataFrame({"facility_id": new_school_index})

new_school_df = new_school_df.merge(
    schools_df, left_on="facility_id", right_on="Unnamed: 0", how="left"
)
new_school_df.head(5)

# %%
# set the parameter
school_indices = range(len(new_time_table[0]))
student_indices = range(len(new_time_table))

# %%
# p-median model considering k nearest facility

from pulp import *
import gurobipy

prob = pulp.LpProblem("k-nearest", LpMinimize)

# create decision variable: whether student i is assigned to school j
decision = LpVariable.dicts(
    "x", ((i, j) for i in student_indices for j in school_indices), 0, 1, LpBinary
)

# set the objective function to minimize the total distance travelled
objective = pulp.lpSum(
    pulp.lpSum(decision[i, j] * new_time_table[i, j] for j in school_indices)
    for i in student_indices
)
prob += objective

# add all the constraints

# 1. Each client is assigned to a facility
for i in student_indices:
    prob += pulp.lpSum(decision[i, j] for j in school_indices) == 1

# 2. Demand value the facility can serve is no more than its capacity.
for j in school_indices:
    prob += (
        pulp.lpSum(decision[i, j] for i in student_indices) <= new_school_df["Count"][j]
    )

# solve the problem
prob.solve(pulp.PULP_CBC_CMD(msg=False))

# %%
for i in student_indices:
    for j in school_indices:
        if decision[i, j].value() == 1:
            print(i, j)

