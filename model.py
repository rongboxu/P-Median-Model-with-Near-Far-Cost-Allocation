#!/usr/bin/env python
# coding: utf-8

# Simple outline for this notebook:
# 
# The title is k-nearest p-median, because this notebook contains the basic code to acheive the Church's model with pulp.
# 
# 1. Test the basic model with k = 5 situation and check the result.
# 
# 2. Test the basic model with k = 1 situation and check the result.
# 
# 3. Add the increasing k value part and make a loop.

# In[ ]:


import pandas as pd
import numpy as np
import pulp
import scipy as sp


# ### k-nearest p-median

# In[3]:


# import data
time_df = pd.read_csv("data/example_subject_student_school_journeys.csv")
students_df = pd.read_csv("data/example_subject_students.csv")
schools_df = pd.read_csv("data/example_subject_schools.csv")


# #### when k = 5, placeholder facility will not be assigned, and the model has optimal solution if only with k nearest facilities

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


# In[5]:


new_time_df = k_smallest_from_distance_table(time_df, "student", "time", "school", 5)
new_time_df


# The second step is to prepare the k-nearest dataframe suitable for the p-median model.   
# 
# We need to create new indexes for client and facilities, and add the capacity information from facility dataframe.

# In[7]:


student_indices = range(new_time_df["student"].nunique())
student_indices


# In[8]:


school_indices = range(new_time_df["facility"].nunique())
school_indices


# In[9]:


# create new index for school/facility, according to their order, and this index is the same with
# that of the array 'time_array' we just created
# to do so, we can easily refer to them in the p-median model
new_time_df["school_new_index"] = (
    new_time_df["facility"].rank(method="dense").astype(int) - 1
)
new_time_df


# In[10]:


# also the new student index
new_time_df["student_new_index"] = (
    new_time_df["student"].rank(method="dense").astype(int) - 1
)
new_time_df


# In[11]:


# in this model, we considerate the existence of capacity, so we add it from the 'schools_df'
new_time_df = new_time_df.merge(
    schools_df[["SE2 PP: Code", "Count"]],
    left_on="facility",
    right_on="SE2 PP: Code",
    how="left",
)
new_time_df


# Now the data has been prepared well.

# In[73]:


def setup_from_travel_table(distance_df, client_indices, facility_indices):
    """
    Using the distance dataframe we prepared
    to write a function that sets up the k-nearest p-median problem. 
    """
    # build the sparse matrix of distance/cost
    # in this matrix, only the distance between clients and k nearest facilities will be stored
    row = distance_df['student_new_index'].values
    col = distance_df['school_new_index'].values
    data = distance_df['time'].values
    sparse_matrix = sp.sparse.csr_array((data, (row, col)))

    # set up the problem
    problem = pulp.LpProblem("k-nearest-p-median", pulp.LpMinimize)

    # set the decision variable for client and k nearest facilities
    decision = pulp.LpVariable.dicts(
        "x",
        (
            (row["student_new_index"], row["school_new_index"])
            for _, row in distance_df.iterrows()
        ),
        0,
        1,
        pulp.LpBinary,
    )

    # set the decision variable for placeholder facility
    decision_g = pulp.LpVariable.dicts("g", (i for i in client_indices), 0, 1, pulp.LpBinary)

    # in order to complete the objective, we need to get the maximum distance for each client
    max_distance = sparse_matrix.max(axis=1).toarray().flatten()

    # set the objective
    objective = pulp.lpSum(
        pulp.lpSum(decision.get((i, j), 0) * sparse_matrix[i, j] for j in facility_indices) + (
            decision_g[i] * (max_distance[i] + 1)
        )
        for i in client_indices
    )
    problem += objective

    # constraint 1. Each client is assigned to a facility
    for i in client_indices:
        problem += pulp.lpSum(decision.get((i, j), 0) for j in facility_indices) + decision_g[i] == 1

    # constraint 2. Demand value the facility can serve is no more than its capacity.
    for j in facility_indices:
        count = distance_df.loc[distance_df["school_new_index"] == j, "Count"].values[0]
        problem += pulp.lpSum(decision.get((i, j), 0) for i in client_indices) <= count

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    return problem, decision, decision_g


# In[74]:


prob, prob_decision, decision_g = setup_from_travel_table(new_time_df, student_indices, school_indices)


# In[75]:


# check if the decision variable is correct
prob_decision


# In[76]:


# also check decision variable for placeholder facility
decision_g


# In[86]:


# check if the problem has optimal solution, if it returns 1, then it has
# if it returns -1, then it has infeasible solution
prob.status


# In[25]:


# print the model result
for i in student_indices:
    for j in school_indices:
        if (i, j) in prob_decision and prob_decision[(i, j)].value() == 1:
            print("student " + str(i) + " is served by school " + str(j))


# In[78]:


# check if any placeholder facility is assigned/selected
# here, no placeholder facility is assigned
for i in student_indices:
    if decision_g[i].value() > 0:
        print("student " + str(i) + " is served by schools far away ")


# #### when k = 1, placeholder facility will be assigned, and the model is infeasible if only with k nearest facilities

# We use the same way as that of k = 5 case to prepare the data.

# In[16]:


new_time_df_k_1 = k_smallest_from_distance_table(time_df, "student", "time", "school", 1)
new_time_df_k_1


# In[17]:


school_indices_k_1 = range(new_time_df_k_1["facility"].nunique())
school_indices_k_1


# In[18]:


new_time_df_k_1["school_new_index"] = (
    new_time_df_k_1["facility"].rank(method="dense").astype(int) - 1
)
new_time_df_k_1["student_new_index"] = (
    new_time_df_k_1["student"].rank(method="dense").astype(int) - 1
)
new_time_df_k_1


# In[19]:


new_time_df_k_1 = new_time_df_k_1.merge(
    schools_df[["SE2 PP: Code", "Count"]],
    left_on="facility",
    right_on="SE2 PP: Code",
    how="left",
)
new_time_df_k_1


# In[79]:


prob_k_1, decision_k_1, decision_k_1_g = setup_from_travel_table(new_time_df_k_1, student_indices, school_indices_k_1)


# In[80]:


prob_k_1.status


# In[81]:


decision_k_1


# In[82]:


decision_k_1_g


# In[83]:


for i in student_indices:
    for j in school_indices:
        if (i, j) in decision_k_1 and decision_k_1[(i, j)].value() == 1:
            print("student " + str(i) + " is served by school " + str(j))


# In[85]:


for i in student_indices:
    if decision_k_1_g[i].value() > 0:
        print("student " + str(i) + " is served by schools far away ")


# From the model results, we can know that:
# 1. The model has optimal solution.
# 2. In the value of `decision_k_1`, `student 1` is missing.
# 3. While, the value of `decision_k_1_g[1]` is more than 1, showing that this placeholder facility is used.

# The next step is finish the step 5 in Levi's message:
# 
# 5. find if a g_i is nonzero. If so, increase the k_i value for that observation and try again.   
# 
# I will work on this part later this week!
