#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pulp import *


# ### k-nearest p-median

# In[6]:


# import data
time_df = pd.read_csv("data/example_subject_student_school_journeys.csv")
students_df = pd.read_csv("data/example_subject_students.csv")
schools_df = pd.read_csv("data/example_subject_schools.csv")


# The first step is to create a new dataframe that only contains the k nearest client-facility pairs from the given dataframe.

# In[4]:


def k_smallest_from_distance_table(
    travel_times, client_name, cost_column, facility_name, k
):
    """
    Given a table of travel times with columns "client", "facility", and "cost",
    make a new DataFrame that contains only the `k` lowest-cost client-facility pairs.
    """
    k_per_client = (
        travel_times.groupby(client_name)[  # for each client
            cost_column
        ]  # look at their "cost" column
        .nsmallest(k)  # and keep the "k" rows with the smallest cost
        .reset_index()  # and reset the index from the groupby
    )
    k_per_client["facility"] = travel_times.loc[
        k_per_client[
            "level_1"
        ],  # look at each row using the row number from the groupby
        facility_name,  # and find the corresponding facility
    ].values  # and ignore the index on the DataFrame
    return k_per_client.drop(
        "level_1", axis=1
    )  # drop the row number column from the groupby


# In[8]:


new_time_df = k_smallest_from_distance_table(time_df, "student", "time", "school", 5)
new_time_df


# In[10]:


# get the array of new distance matrix from new dataframe
time_array = (
    new_time_df.pivot_table(
        columns="facility",
        fill_value=10000,
        index="student",
        sort=False,
        values="time",
    )
    .astype(int)
    .values
)
time_array[:5]


# The second step is to prepare the k-nearest dataframe suitable for the p-median model.   
# 
# We need to create new indexes for client and facilities, and add the capacity information from facility dataframe.

# In[14]:


student_indices = range(new_time_df["student"].nunique())
student_indices


# In[15]:


school_indices = range(new_time_df["facility"].nunique())
school_indices


# In[16]:


# create new index for school/facility, according to their order, and this index is the same with
# that of the array 'time_array' we just created
# to do so, we can easily refer to them in the p-median model
new_time_df["school_new_index"] = (
    new_time_df["facility"].rank(method="dense").astype(int) - 1
)
new_time_df


# In[17]:


# also the new student index
new_time_df["student_new_index"] = (
    new_time_df["student"].rank(method="dense").astype(int) - 1
)
new_time_df


# In[30]:


# in this model, we considerate the existence of capacity, so we add it from the 'schools_df'
new_time_df = new_time_df.merge(
    schools_df[["SE2 PP: Code", "Count"]],
    left_on="facility",
    right_on="SE2 PP: Code",
    how="left",
)
new_time_df


# Now the data has been prepared well.

# In[33]:


def setup_from_travel_table(distance_df):
    """
    Using the distance dataframe we prepared
    to write a function that sets up the k-nearest p-median problem. 
    """
    # convert the dataframe to the 2D array
    time_array = (
    distance_df.pivot_table(
        columns="facility",
        fill_value=10000,
        index="student",
        sort=False,
        values="time",
    )
    .astype(int)
    .values
    )

    # set up the problem
    problem = pulp.LpProblem("k-nearest-p-median", LpMinimize)

    decision = LpVariable.dicts(
        "x",
        (
            (row["student_new_index"], row["school_new_index"])
            for _, row in distance_df.iterrows()
        ),
        0,
        1,
        LpBinary,
    )

    objective = pulp.lpSum(
        pulp.lpSum(decision.get((i, j), 0) * time_array[i, j] for j in school_indices)
        for i in student_indices
    )
    problem += objective

    # 1. Each client is assigned to a facility
    for i in student_indices:
        problem += pulp.lpSum(decision.get((i, j), 0) for j in school_indices) == 1

    # 2. Demand value the facility can serve is no more than its capacity.
    for j in school_indices:
        count = distance_df.loc[distance_df["school_new_index"] == j, "Count"].values[0]
        problem += pulp.lpSum(decision.get((i, j), 0) for i in student_indices) <= count

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    return problem, decision


# In[34]:


prob, prob_decision = setup_from_travel_table(new_time_df)


# In[36]:


# check if the decision variable is correct
prob_decision


# In[35]:


for i in student_indices:
    for j in school_indices:
        if (i, j) in prob_decision and prob_decision[(i, j)].value() == 1:
            print("student " + str(i) + " is served by school " + str(j))

