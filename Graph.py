# -*- coding: utf-8 -*-


#Regions and Season
clarity_color_table = pd.crosstab(index=df_fn["Regions"], 
                          columns=df_fn["Season"])

clarity_color_table


clarity_color_table.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=True)

#Regions and Premise type
clarity_color_table = pd.crosstab(index=df_fn["Premise_Type"], 
                          columns=df_fn["Regions"])

clarity_color_table


clarity_color_table.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=True)
#Season and week
clarity_color_table = pd.crosstab(index=df_fn["Week"], 
                          columns=df_fn["Season"])

clarity_color_table


clarity_color_table.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=True)
#color and bike type
clarity_color_table = pd.crosstab(index=df_fn["BikeType"],
                                  columns=df_fn["Color"])

clarity_color_table


clarity_color_table.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=True)
#Countplots for each of our categorical variables.
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for variable, subplot in zip(newCol, ax.flatten()):
    sns.countplot(df_fn[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
