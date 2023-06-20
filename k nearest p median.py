#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pulp import *


# ### the p-median model with k nearest facilities

# For this first stage, only k nearest facilities will be imported in the model.    
# 1. After importing original data, I create the new table only containing k nearest facilities.   
# 2. To test the method, I use the test IOE data as I used for capacitated p-median case.   
# So in this model, the capacity constraint is still live.   
# 3. In a broader case, it should be that, the sum of demand value of the facility can serve is no more than its capacity.

# In[2]:


# import data
time = pd.read_csv('data/example_subject_student_school_journeys.csv')
time_table = (
    time.pivot_table(
        columns="school",
        fill_value=10000,
        index="student",
        sort=False,
        values="time",
    )
    .astype(int)
    .values
)
students_df = pd.read_csv('data/example_subject_students.csv')
schools_df = pd.read_csv('data/example_subject_schools.csv')


# In[3]:


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
        row = {'client_id': client_idx, 'facility_id': i, 'distance': distances[i]}
        rows.append(row)

# Create a new_distance_df dataframe by concatenating the rows
new_distance_df = pd.DataFrame(rows, columns=['client_id', 'facility_id', 'distance'])


# In[4]:


new_distance_df.head(10)


# In[5]:


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


# In[6]:


new_time_table[:4]


# In[7]:


# create capacity df
new_school_index =  new_distance_df['facility_id'].sort_values().unique()
new_school_df = pd.DataFrame({'facility_id': new_school_index})

new_school_df = new_school_df.merge(schools_df, left_on='facility_id',right_on='Unnamed: 0', how='left')
new_school_df.head(5)


# In[8]:


# set the parameter
school_indices = range(len(new_time_table[0]))
student_indices = range(len(new_time_table))


# In[9]:


# p-median model considering k nearest facility

from pulp import *
import gurobipy

prob = pulp.LpProblem("k-nearest", LpMinimize)

# create decision variable: whether student i is assigned to school j
decision = LpVariable.dicts("x", ((i, j) for i in student_indices for j in school_indices), 0, 1, LpBinary)

# set the objective function to minimize the total distance travelled
objective = pulp.lpSum(
    pulp.lpSum(decision[i,j] * new_time_table[i,j] for j in school_indices) 
    for i in student_indices)
prob += objective

# add all the constraints

# 1. Each client is assigned to a facility
for i in student_indices:
    prob +=  pulp.lpSum(decision[i,j] for j in school_indices) == 1

# 2. Demand value the facility can serve is no more than its capacity.
for j in school_indices:
    prob +=  pulp.lpSum(decision[i,j] for i in student_indices) <= new_school_df['Count'][j]
    
# solve the problem
prob.solve(pulp.PULP_CBC_CMD(msg=False))


# In[10]:


for i in student_indices:
    for j in school_indices:
        if decision[i,j].value() == 1:
            print(i,j)


# ### second stage: find the infeasible K and implement the extra facility

# In[11]:


# define the function to get the new distance matrix only with k nearest facilities
def get_k_facilities(k, distance_array):

    # Create an empty list to hold the rows
    rows = []

    # Iterate over each client point and add the k nearest facility indices to the new pivot table
    for client_idx in range(distance_array.shape[0]):
        distances = distance_array[client_idx]
        nearest_index = np.argsort(distances)[:k]
        # Append rows to the list
        for i in nearest_index:
            row = {'client_id': client_idx, 'facility_id': i, 'distance': distances[i]}
            rows.append(row)

    # Create a new_distance_df dataframe by concatenating the rows
    new_distance_df = pd.DataFrame(rows, columns=['client_id', 'facility_id', 'distance'])

    # create new distance matrix
    new_distance_array = (
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

    return new_distance_df, new_distance_array


# In[12]:


# define the function to get the capacity df
def get_capacity(new_distance_df,schools_df):

    new_school_index = new_distance_df['facility_id'].sort_values().unique()
    new_school_df = pd.DataFrame({'facility_id': new_school_index})

    new_school_df = new_school_df.merge(schools_df, left_on='facility_id',right_on='Unnamed: 0', how='left')
    return new_school_df


# #### try k = 2

# In[13]:


# try k = 2
distance_2_df, distance_2_arr = get_k_facilities(2, time_table)
school_2_df = get_capacity(distance_2_df, schools_df)


# In[14]:


# p-median model

prob_2 = pulp.LpProblem("k-nearest-2", LpMinimize)

# set the parameter
school_2_indices = range(len(distance_2_arr[0]))

# create decision variable: whether student i is assigned to school j
decision_2 = LpVariable.dicts("x", ((i, j) for i in student_indices for j in school_2_indices), 0, 1, LpBinary)

# set the objective function to minimize the total distance travelled
objective_2 = pulp.lpSum(
    pulp.lpSum(decision_2[i,j] * distance_2_arr[i,j] for j in school_2_indices) 
    for i in student_indices)
prob_2 += objective_2

# add all the constraints
# 1. Each client is assigned to a facility
for i in student_indices:
    prob_2 +=  pulp.lpSum(decision_2[i,j] for j in school_2_indices) == 1

# 2. Demand value the facility can serve is no more than its capacity.
for j in school_2_indices:
    prob_2 +=  pulp.lpSum(decision_2[i,j] for i in student_indices) <= school_2_df['Count'][j]
    
# solve the problem
prob_2.solve(pulp.PULP_CBC_CMD(msg=False))


# In[15]:


distance_2_df


# In[16]:


for i in student_indices:
    for j in school_2_indices:
        if decision_2[i,j].value() == 1:
            print(i,j)


# So, k = 2 has the optimal solution. k = 1 should have the infeasible result.

# #### try k = 1

# In[17]:


distance_1_df, distance_1_arr = get_k_facilities(1, time_table)
school_1_df = get_capacity(distance_1_df, schools_df)

# p-median
prob_1 = pulp.LpProblem("k-nearest-1", LpMinimize)
school_1_indices = range(len(distance_1_arr[0]))

decision_1 = LpVariable.dicts("x", ((i, j) for i in student_indices for j in school_1_indices), 0, 1, LpBinary)

objective_1 = pulp.lpSum(
    pulp.lpSum(decision_1[i,j] * distance_1_arr[i,j] for j in school_1_indices) 
    for i in student_indices)
prob_1 += objective_1

# 1. Each client is assigned to a facility
for i in student_indices:
    prob_1 +=  pulp.lpSum(decision_1[i,j] for j in school_1_indices) == 1

# 2. Demand value the facility can serve is no more than its capacity.
for j in school_1_indices:
    prob_1 +=  pulp.lpSum(decision_1[i,j] for i in student_indices) <= school_1_df['Count'][j]
    
prob_1.solve(pulp.PULP_CBC_CMD(msg=False))


# In[18]:


for i in student_indices:
    for j in school_1_indices:
        if decision_1[i,j].value() == 1:
            print(i,j)


# In[19]:


school_1_index =  distance_1_df['facility_id'].sort_values().unique()
school_1_index


# In[20]:


school_1_index[2]


# In[21]:


distance_1_df


# In[22]:


distance_1_arr


# In[23]:


print(round(prob_1.objective.value(), 3))


# The problem is that it still takes one 10000 as the result, which is not allowed.   
# 
# Here my solution is to add one constraint, to prevent the use of 10000 in the solution.

# In[24]:


# p-median with constraint to prevent the use of 10000
prob_1_new = pulp.LpProblem("k-nearest-1-new-constraint", LpMinimize)

decision_1_prevent = LpVariable.dicts("x", ((i, j) for i in student_indices for j in school_1_indices), 0, 1, LpBinary)

objective_1_prevent = pulp.lpSum(
    pulp.lpSum(decision_1_prevent[i,j] * distance_1_arr[i,j] for j in school_1_indices) 
    for i in student_indices)
prob_1_new += objective_1_prevent

# 1. Each client is assigned to a facility
for i in student_indices:
    prob_1_new +=  pulp.lpSum(decision_1_prevent[i,j] for j in school_1_indices) == 1

# 2. Demand value the facility can serve is no more than its capacity.
for j in school_1_indices:
    prob_1_new +=  pulp.lpSum(decision_1_prevent[i,j] for i in student_indices) <= school_1_df['Count'][j]

# 3. To prevent the use of 10000
for i in student_indices:
    for j in school_1_indices:
        if distance_1_arr[i,j] == 10000:
            prob_1_new += decision_1_prevent[i,j] == 0

prob_1_new.solve(pulp.PULP_CBC_CMD(msg=False))


# #### add decision variable for placeholder/extra facility

# Since k = 1 is infeasible, we use this case to implement the extra facility.

# In[25]:


# get the new distance matrix with k nearest facilities, and the rest facilities


# In[26]:


k_facility_number = len(distance_1_df['facility_id'].unique())
print(k_facility_number)


# In[27]:


distance_1_df['facility_id']


# In[28]:


# Get the complement of facility_id in distance_1_df
extra_facility_ids = [i for i in range(len(time_table[0])) if i not in distance_1_df['facility_id'].values]
extra_facility_ids = np.array(extra_facility_ids)
extra_facility_ids


# In[29]:


len(extra_facility_ids)


# In[30]:


len(time_table[0])


# In[31]:


extra_distance_matrix = time_table[:, extra_facility_ids]
extra_distance_matrix[0]


# In[32]:


# get the capacity of extra facilities
school_extra_df = pd.DataFrame({'facility_id': extra_facility_ids})
school_extra_df = school_extra_df.merge(schools_df, left_on='facility_id',right_on='Unnamed: 0', how='left')


# In[33]:


# create the new distance matrix of k nearest facilities
# in the previous one, the non-k-nearest facilities have 10,000 as the distance
# now we need to use the real distance
facility_ids_array = distance_1_df['facility_id'].sort_values().values
k_distance_matrix = time_table[:, facility_ids_array]
k_distance_matrix


# In[34]:


facility_ids_array


# In[35]:


distance_1_arr


# In[36]:


# p-median with all the facilities
prob_all = pulp.LpProblem("k-nearest-all-facilities", LpMinimize)

# set parameter
school_extra_indices = range(len(extra_distance_matrix[0]))
decision_extra = LpVariable.dicts("x_extra", ((i, j) for i in student_indices for j in school_extra_indices), 0, 1, LpBinary)
decision_k_in_all = LpVariable.dicts("x_k", ((i, j) for i in student_indices for j in school_1_indices), 0, 1, LpBinary)

# set objective
objective_all = (
    pulp.lpSum(
        pulp.lpSum(decision_k_in_all[i, j] * k_distance_matrix[i, j] for j in school_1_indices)
        for i in student_indices
    ) +
    pulp.lpSum(
        pulp.lpSum(decision_extra[i, j] * extra_distance_matrix[i, j] for j in school_extra_indices)
        for i in student_indices
    )
)
prob_all += objective_all

# 1. Each client is assigned to a facility
for i in student_indices:
    prob_all += (
        pulp.lpSum(decision_k_in_all[i, j] for j in school_1_indices)
        + pulp.lpSum(decision_extra[i, j] for j in school_extra_indices)
        == 1
    )

# 2. Demand value the facility can serve is no more than its capacity.
for j in school_1_indices:
    prob_all +=  pulp.lpSum(decision_k_in_all[i,j] for i in student_indices) <= school_1_df['Count'][j]
for j in school_extra_indices:
    prob_all +=  pulp.lpSum(decision_extra[i,j] for i in student_indices) <= school_extra_df['Count'][j]

#prob_all.solve(pulp.PULP_CBC_CMD(msg=False))
prob_all.solve(GLPK(msg=False))


# In[37]:


prob_all.status


# status = 1: The problem was solved to optimality, and an optimal solution was found.   
# status = 0: The problem is feasible, but the solver was not able to prove that the solution is optimal.   
# status = -1: The problem is infeasible; no feasible solution exists.   
# status = -2: The problem is unbounded; the objective function can be improved infinitely.   
# status = -3: The solver encountered an error or was unable to solve the problem.    

# In[41]:


for (i, j) in decision_extra.keys():
    value = decision_extra[(i, j)].varValue
    if value == 1:
        print("decision_extra[{}, {}] = {}".format(i, j, value))

    # Access the optimal values of decision_k_in_all
for (i, j) in decision_k_in_all.keys():
    value = decision_k_in_all[(i, j)].varValue
    if value == 1:
        print("decision_k_in_all[{}, {}] = {}".format(i, j, value))


# Another condition is when do extra facility is selected, but the k-nearest facility of other client is selected.   
# 
# For example, when k = 2, client `a` has facility `1` and `2` as the nearest facility, client `b` has facility `3` and `4`.   
# 
# It's possible that in the optimal solution client `b` is assigned to facility `1`, which is not its nearest facility.
