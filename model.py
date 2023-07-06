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

# In[1]:


import pandas as pd
import numpy as np
import pulp
import scipy as sp


# ### k-nearest p-median

# In[2]:


# import data
time_df = pd.read_csv("data/example_subject_student_school_journeys.csv")
students_df = pd.read_csv("data/example_subject_students.csv")
schools_df = pd.read_csv("data/example_subject_schools.csv")


# #### when k = 5, placeholder facility will not be assigned, and the model has optimal solution if only with k nearest facilities

# The first step is to create a new dataframe that only contains the k nearest client-facility pairs from the given dataframe.

# In[3]:


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


# In[4]:


new_time_df = k_smallest_from_distance_table(time_df, "student", "time", "school", 5)
new_time_df


# The second step is to prepare the k-nearest dataframe suitable for the p-median model.   
# 
# We need to create new indexes for client and facilities, and add the capacity information from facility dataframe.

# In[5]:


student_indices = range(new_time_df["student"].nunique())
student_indices


# In[6]:


school_indices = range(new_time_df["facility"].nunique())
school_indices


# In[7]:


# create new index for school/facility, according to their order, and this index is the same with
# that of the array 'time_array' we just created
# to do so, we can easily refer to them in the p-median model
new_time_df["school_new_index"] = (
    new_time_df["facility"].rank(method="dense").astype(int) - 1
)
new_time_df


# In[8]:


# also the new student index
new_time_df["student_new_index"] = (
    new_time_df["student"].rank(method="dense").astype(int) - 1
)
new_time_df


# In[9]:


# in this model, we considerate the existence of capacity, so we add it from the 'schools_df'
new_time_df = new_time_df.merge(
    schools_df[["SE2 PP: Code", "Count"]],
    left_on="facility",
    right_on="SE2 PP: Code",
    how="left",
)
new_time_df


# Now the data has been prepared well.

# In[10]:


def setup_from_travel_table(distance_df, client_indices, facility_indices):
    """
    Using the distance dataframe we prepared
    to write a function that sets up the k-nearest p-median problem. 
    """
    # build the sparse matrix of distance/cost
    # in this matrix, only the distance between clients and k nearest facilities will be stored
    row = distance_df["student_new_index"].values
    col = distance_df["school_new_index"].values
    data = distance_df["time"].values
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
    decision_g = pulp.LpVariable.dicts(
        "g", (i for i in client_indices), 0, 1, pulp.LpBinary
    )

    # in order to complete the objective, we need to get the maximum distance for each client
    max_distance = sparse_matrix.max(axis=1).toarray().flatten()

    # set the objective
    objective = pulp.lpSum(
        pulp.lpSum(
            decision.get((i, j), 0) * sparse_matrix[i, j] for j in facility_indices
        )
        + (decision_g[i] * (max_distance[i] + 1))
        for i in client_indices
    )
    problem += objective

    # constraint 1. Each client is assigned to a facility
    for i in client_indices:
        problem += (
            pulp.lpSum(decision.get((i, j), 0) for j in facility_indices)
            + decision_g[i]
            == 1
        )

    # constraint 2. Demand value the facility can serve is no more than its capacity.
    for j in facility_indices:
        count = distance_df.loc[distance_df["school_new_index"] == j, "Count"].values[0]
        problem += pulp.lpSum(decision.get((i, j), 0) for i in client_indices) <= count

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    return problem, decision, decision_g


# In[11]:


prob, prob_decision, decision_g = setup_from_travel_table(
    new_time_df, student_indices, school_indices
)



# In[12]:


# check if the decision variable is correct
prob_decision


# In[13]:


# also check decision variable for placeholder facility
decision_g


# In[14]:


# check if the problem has optimal solution, if it returns 1, then it has
# if it returns -1, then it has infeasible solution
prob.status


# In[15]:


# print the model result
for i in student_indices:
    for j in school_indices:
        if (i, j) in prob_decision and prob_decision[(i, j)].value() == 1:
            print("student " + str(i) + " is served by school " + str(j))


# In[16]:


# check if any placeholder facility is assigned/selected
# here, no placeholder facility is assigned
for i in student_indices:
    if decision_g[i].value() > 0:
        print("student " + str(i) + " is served by schools far away ")


# #### when k = 1, placeholder facility will be assigned, and the model is infeasible if only with k nearest facilities

# We use the same way as that of k = 5 case to prepare the data.

# In[17]:


new_time_df_k_1 = k_smallest_from_distance_table(
    time_df, "student", "time", "school", 1
)
new_time_df_k_1


# In[18]:


school_indices_k_1 = range(new_time_df_k_1["facility"].nunique())
school_indices_k_1


# In[19]:


new_time_df_k_1["school_new_index"] = (
    new_time_df_k_1["facility"].rank(method="dense").astype(int) - 1
)
new_time_df_k_1["student_new_index"] = (
    new_time_df_k_1["student"].rank(method="dense").astype(int) - 1
)
new_time_df_k_1


# In[20]:


new_time_df_k_1 = new_time_df_k_1.merge(
    schools_df[["SE2 PP: Code", "Count"]],
    left_on="facility",
    right_on="SE2 PP: Code",
    how="left",
)
new_time_df_k_1


# In[21]:


prob_k_1, decision_k_1, decision_k_1_g = setup_from_travel_table(
    new_time_df_k_1, student_indices, school_indices_k_1
)



# In[22]:


prob_k_1.status


# In[23]:


decision_k_1


# In[24]:


decision_k_1_g


# In[25]:


for i in student_indices:
    for j in school_indices:
        if (i, j) in decision_k_1 and decision_k_1[(i, j)].value() == 1:
            print("student " + str(i) + " is served by school " + str(j))


# In[26]:


for i in student_indices:
    if decision_k_1_g[i].value() > 0:
        print("student " + str(i) + " is served by schools far away ")


# From the model results, we can know that:
# 1. The model has optimal solution.
# 2. In the value of `decision_k_1`, `student 1` is missing.
# 3. While, the value of `decision_k_1_g[1]` is more than 1, showing that this placeholder facility is used.

# #### If any g_i is nonzero, increase the k_i value for that observation and try again. 

# In[28]:


# check if any g_i is nonzero, and increase the k value for client i
# create the new k value list
k_replace = [1] * len(student_indices)
for i in student_indices:
    if decision_k_1_g[i].value() > 0:
        k_replace[i] = 2


# In[29]:


k_replace


# In[32]:


# the first way is to create a new dataframe, and import it into the model
# this way will 'restart' the model every time
def recreate_k_smallest_from_distance_table(
    travel_times, client_name, cost_column, facility_name, k_list
):
    result = pd.DataFrame()  # Create an empty DataFrame to store the results

    for client, k in zip(travel_times[client_name].unique(), k_list):
        k_per_client = (
            travel_times[travel_times[client_name] == client]
            .nsmallest(k, cost_column)
            .reset_index(drop=True)
        )
        result = pd.concat([result, k_per_client], ignore_index=True)

    result["facility"] = result[facility_name]
    result = result.drop(columns=[facility_name])

    return result


# In[49]:


new_time_df_k_1_list = recreate_k_smallest_from_distance_table(
    time_df, "student", "time", "school", k_replace
)
new_time_df_k_1_list


# In[50]:


# prepare the data like the previous steps
school_indices_k_1_list = range(new_time_df_k_1_list["facility"].nunique())

new_time_df_k_1_list["school_new_index"] = (
    new_time_df_k_1_list["facility"].rank(method="dense").astype(int) - 1
)
new_time_df_k_1_list["student_new_index"] = (
    new_time_df_k_1_list["student"].rank(method="dense").astype(int) - 1
)

new_time_df_k_1_list = new_time_df_k_1_list.merge(
    schools_df[["SE2 PP: Code", "Count"]],
    left_on="facility",
    right_on="SE2 PP: Code",
    how="left",
)
new_time_df_k_1_list


# In[51]:


prob_k_1_list, decision_k_1_list, decision_k_1_g_list = setup_from_travel_table(
    new_time_df_k_1_list, student_indices, school_indices_k_1_list
)



# In[52]:


prob_k_1_list.status


# In[53]:


decision_k_1_list


# In[54]:


for i in student_indices:
    if decision_k_1_g_list[i].value() > 0:
        print("student " + str(i) + " is served by schools far away ")


# In[56]:


for i in student_indices:
    for j in school_indices_k_1_list:
        if (i, j) in decision_k_1_list and decision_k_1_list[(i, j)].value() == 1:
            print("student " + str(i) + " is served by school " + str(j))


# The result shows that 'restarting' the model may bring the unexpected outcomes.   
# 
# In the `k = 1` model previously, student 1 is assigned to the faraway facility, and student 8 is assigned to the facility with the time of 79 minutes. This is because student 1 and student 8 have the same nearest facility, and that facility can only accommodate one student.   
# 
# In the `new k = 1` model, which we increase the k value to 2 for student 1, the result shows student 1 is assigned to its nearest facility, while student 8 is assigned to the faraway facility.   
# 
# The next step I think about is either trying to write model which can be `resolved`, or continuing to increase the k value for student 8 for `restarting`.

# ##### increase k_i again

# In[57]:


# check if any g_i is nonzero, and increase the k value for client i
# create the new k value list
k_replace_2 = k_replace
for i in student_indices:
    if decision_k_1_g_list[i].value() > 0:
        k_replace_2[i] = k_replace[i] + 1
k_replace_2


# In[58]:


new_time_df_k_1_list_2 = recreate_k_smallest_from_distance_table(
    time_df, "student", "time", "school", k_replace_2
)
new_time_df_k_1_list_2


# In[59]:


# prepare the data like the previous steps
school_indices_k_1_list_2 = range(new_time_df_k_1_list_2["facility"].nunique())

new_time_df_k_1_list_2["school_new_index"] = (
    new_time_df_k_1_list_2["facility"].rank(method="dense").astype(int) - 1
)
new_time_df_k_1_list_2["student_new_index"] = (
    new_time_df_k_1_list_2["student"].rank(method="dense").astype(int) - 1
)

new_time_df_k_1_list_2 = new_time_df_k_1_list_2.merge(
    schools_df[["SE2 PP: Code", "Count"]],
    left_on="facility",
    right_on="SE2 PP: Code",
    how="left",
)
new_time_df_k_1_list_2


# In[60]:


prob_k_1_list_2, decision_k_1_list_2, decision_k_1_g_list_2 = setup_from_travel_table(
    new_time_df_k_1_list_2, student_indices, school_indices_k_1_list_2
)



# In[62]:


for i in student_indices:
    if decision_k_1_g_list_2[i].value() > 0:
        print("student " + str(i) + " is served by schools far away ")


# In[63]:


for i in student_indices:
    for j in school_indices_k_1_list_2:
        if (i, j) in decision_k_1_list_2 and decision_k_1_list_2[(i, j)].value() == 1:
            print("student " + str(i) + " is served by school " + str(j))


# In this model, no faraway facility is used.

# #### Set up the iteration for increasing k value to make sure no faraway facility is assigned

# In[74]:


# define 4 functions for the iteration


def create_k_nearest_dataframe(
    distance, client_name, cost_column, facility_name, k_list
):
    """
    create the dataframe contains the distance between the clients and their k nearest facilities;
    """
    result = pd.DataFrame()

    for client, k in zip(distance[client_name].unique(), k_list):
        k_per_client = (
            distance[distance[client_name] == client]
            .nsmallest(k, cost_column)
            .reset_index(drop=True)
        )
        result = pd.concat([result, k_per_client], ignore_index=True)

    result.rename(
        columns={
            client_name: "client",
            facility_name: "facility",
            cost_column: "distance",
        },
        inplace=True,
    )

    # create new index for clients and facilities
    result["facility_new_index"] = (
        result["facility"].rank(method="dense").astype(int) - 1
    )
    result["client_new_index"] = result["client"].rank(method="dense").astype(int) - 1

    return result


def add_facility_capacity(distance, facility_df, facility_name, capacity_column):
    """
    add the facility capacity information to the distance dataframe
    """
    distance = distance.merge(
        facility_df[[facility_name, capacity_column]],
        left_on="facility",
        right_on=facility_name,
        how="left",
    )
    distance.rename(columns={capacity_column: "capacity"}, inplace=True)
    return distance


def set_up_allocation_model(distance):
    """
    use the distance dataframe we prepared
    to write a function that sets up the k-nearest p-median problem. 
    """

    # build the sparse matrix of distance/cost
    # in this matrix, only the distance between clients and k nearest facilities will be stored
    row = distance["client_new_index"].values
    col = distance["facility_new_index"].values
    data = distance["distance"].values
    sparse_matrix = sp.sparse.csr_array((data, (row, col)))

    # create the indices for clients and facilities
    client_indices = range(0, distance["client_new_index"].max() + 1)
    facility_indices = range(0, distance["facility_new_index"].max() + 1)

    # set up the model
    problem = pulp.LpProblem("k-nearest-p-median", pulp.LpMinimize)

    # set the decision variable for client and k nearest facilities
    decision = pulp.LpVariable.dicts(
        "x",
        (
            (row["client_new_index"], row["facility_new_index"])
            for _, row in distance.iterrows()
        ),
        0,
        1,
        pulp.LpBinary,
    )

    # set the decision variable for placeholder facility
    decision_g = pulp.LpVariable.dicts(
        "g", (i for i in client_indices), 0, 1, pulp.LpBinary
    )

    # set the objective
    # to complete the objective, we need to get the maximum distance for each client first
    max_distance = sparse_matrix.max(axis=1).toarray().flatten()
    objective = pulp.lpSum(
        pulp.lpSum(
            decision.get((i, j), 0) * sparse_matrix[i, j] for j in facility_indices
        )
        + (decision_g[i] * (max_distance[i] + 1))
        for i in client_indices
    )
    problem += objective

    # constraint 1. Each client is assigned to a facility
    for i in client_indices:
        problem += (
            pulp.lpSum(decision.get((i, j), 0) for j in facility_indices)
            + decision_g[i]
            == 1
        )

    # constraint 2. Demand value the facility can serve is no more than its capacity.
    for j in facility_indices:
        capacity = distance.loc[distance["facility_new_index"] == j, "capacity"].values[
            0
        ]
        problem += (
            pulp.lpSum(decision.get((i, j), 0) for i in client_indices) <= capacity
        )

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    return problem, decision, decision_g


def create_k_list(decision_g, k_list):
    """
    increase the k value of client with the g_i > 0, create a new k list
    """
    new_k_list = k_list
    for i in range(len(decision_g)):
        if decision_g[i].value() > 0:
            new_k_list[i] = new_k_list[i] + 1
    return new_k_list


# In[86]:


k_value = 1  # Set the initial k value based on user input
k_list = [k_value] * len(range(time_df["student"].nunique()))
sum_gi = 1

while sum_gi > 0:
    distance_df = create_k_nearest_dataframe(
        time_df, "student", "time", "school", k_list
    )
    distance_df = add_facility_capacity(
        distance_df, schools_df, "SE2 PP: Code", "Count"
    )
    prob, decision, decision_g = set_up_allocation_model(distance_df)

    sum_gi = 0
    for i in range(len(decision_g)):
        sum_gi += decision_g[i].value()

    if sum_gi > 0:
        k_list = create_k_list(decision_g, k_list)


# In[87]:


k_list


# In[92]:


print(pulp.value(prob.objective))


# In[88]:


for i in student_indices:
    for j in range(0, distance_df["facility_new_index"].max() + 1):
        if (i, j) in decision and decision[(i, j)].value() == 1:
            facility_id = distance_df.loc[
                distance_df["facility_new_index"] == j, "facility"
            ].values[0]
            print("student " + str(i) + " is served by school " + facility_id)


# In[93]:


k_value = 5  # Set the initial k value based on user input
k_list = [k_value] * len(range(time_df["student"].nunique()))
sum_gi = 1

while sum_gi > 0:
    distance_df = create_k_nearest_dataframe(
        time_df, "student", "time", "school", k_list
    )
    distance_df = add_facility_capacity(
        distance_df, schools_df, "SE2 PP: Code", "Count"
    )
    prob, decision, decision_g = set_up_allocation_model(distance_df)

    sum_gi = 0
    for i in range(len(decision_g)):
        sum_gi += decision_g[i].value()

    if sum_gi > 0:
        k_list = create_k_list(decision_g, k_list)


# In[90]:


k_list


# In[94]:


print(pulp.value(prob.objective))


# In[85]:


for i in student_indices:
    for j in range(0, distance_df["facility_new_index"].max() + 1):
        if (i, j) in decision and decision[(i, j)].value() == 1:
            facility_id = distance_df.loc[
                distance_df["facility_new_index"] == j, "facility"
            ].values[0]
            print("student " + str(i) + " is served by school " + facility_id)

