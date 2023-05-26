import numpy as np
import pandas
import math
import matplotlib.pyplot as plt
import time


def run_model(age,weight,height,budget,pref_ingredients,allergies,activity,sex):
    alpha = 0.05
    num_episodes = 500
    epsilon = 0.5
    breakfast = pandas.read_csv('breakfast.csv', encoding='utf-8', delimiter=";")
    lunch = pandas.read_csv('lunch.csv', encoding='utf-8', delimiter=";")
    dinner = pandas.read_csv('dinner.csv', encoding='utf-8', delimiter=";")
    meals = pandas.concat([breakfast, lunch, dinner])
    data = meals[['Recipe ID', 'Meal', 'Q_Merge_Label', 'Calories', 'Protein', 'Carbs', 'Fat', 'Price', "V_0"]]
    reward = Individual_Reward(allergies,pref_ingredients, meals)
    calories,protein,fat,carbs=nutrition_calculator(weight, height, activity, sex, age,)
    result_model=Model(data,alpha,num_episodes,epsilon,budget,reward,fat,carbs,protein)
    best_breakfast,best_lunch,best_dinner=get_recipe_values(result_model,meals)
    return best_breakfast,best_lunch,best_dinner



def Individual_Reward(allergens_list,preference_list,meals):
    recipes = np.array(meals["Ingredients"])
    reward = np.zeros(len(recipes))
    index = 0
    for recipe in recipes:
        no_ingredients = 0
        for preference in preference_list:
            for ingredient in eval(recipe):
                if (preference in ingredient):
                    no_ingredients += 1
                    reward[index] += 2*math.log(3 * no_ingredients + 1)
        for allergen in allergens_list:
            for ingredient in eval(recipe):
                if (allergen in ingredient):
                    reward[index] = -10
        index += 1
    print(reward)
    return reward

def Error(theo_value,practical_value):
    return 1-abs(theo_value-practical_value)/theo_value


def Model(data, alpha, e, epsilon, budget, reward, fat, carbs, protein):
    # Define the States
    Recipies = list(set(data['Meal']))
    # Initialise V_0
    V0 = data['V_0']
    data.loc[:, 'V'] = V0
    output = []
    output1 = []
    output2 = []
    actioninfull = []
    # Repeat for the number of episodes
    for e in range(0, e):

        episode_run = []
        # Introduce epsilon-greedy selection, we randomly select the first episode as V_0(a) = 0 for all actions
        epsilon = epsilon
        if e == 0:
            for i in range(0, len(Recipies)):
                episode_run = np.append(episode_run, np.random.randint(1,10, size=None))
            episode_run = episode_run.astype(int)

        else:
            for i in range(0, len(Recipies)):
                greedyselection = np.random.randint(1, 10)
                if greedyselection <= (epsilon) * 10:
                    episode_run = np.append(episode_run, np.random.randint(1, 10, size=None))
                else:
                    data_I = data[data['Meal'] == (i+1)]
                    MaxofVforI = data_I[data_I['V'] == data_I['V'].max()]['Recipe ID']
                    # If multiple max values, take first
                    MaxofVforI = MaxofVforI.values[0]
                    episode_run = np.append(episode_run, MaxofVforI)

                episode_run = episode_run.astype(int)

        episode = pandas.DataFrame({'Meal': Recipies, 'Recipe ID': episode_run})
        episode['Merged_label'] = (episode['Meal'] * 10 + episode['Recipe ID']).astype(float)
        data.loc[:,'Q_Merge_Label'] = (data['Q_Merge_Label']).astype(float)
        data.loc[:,'Reward'] = reward
        episode2 = episode.merge(data[['Q_Merge_Label', 'Price','Calories','Protein','Carbs','Fat', 'Reward']], left_on='Merged_label',
                                 right_on='Q_Merge_Label', how='inner')
        data = data.drop('Reward', 1)

        # Calculate our terminal reward
        if (budget >= episode2['Price'].sum()):
            Return = 5 + Error(carbs,episode2['Carbs'].sum())+Error(protein,episode2['Protein'].sum())+Error(fat,episode2['Fat'].sum()) +episode2['Reward'].sum()
        else:
            Return = -5 + Error(carbs,episode2['Carbs'].sum())+Error(protein,episode2['Protein'].sum())+Error(fat,episode2['Fat'].sum()) +episode2['Reward'].sum()
        episode2 = episode2.drop('Reward', 1)
        episode2['Return'] = Return

        # Apply update rule to actions that were involved in obtaining terminal reward
        data = data.merge(episode2[['Merged_label', 'Return']], left_on='Q_Merge_Label', right_on='Merged_label',
                          how='outer')
        data['Return'] = data['Return'].fillna(0)
        for v in range(0, len(data)):
            if data.iloc[v, 11] == 0:
                data.iloc[v, 9] = data.iloc[v, 9]
            else:
                data.iloc[v, 9] = data.iloc[v, 9] + alpha * ((data.iloc[v, 11] / len(Recipies)) - data.iloc[v, 9])

        # Output table
        data = data.drop('Merged_label', 1)
        data = data.drop('Return', 1)

        # Output is the Sum of V(a) for all episodes
        output = np.append(output, data.iloc[:, -1].sum())


        # Ouput to optimal action from the model based on highest V(a)
        action = pandas.DataFrame(data.groupby('Meal')['V'].max())
        action2 = action.merge(data, left_on='V', right_on='V', how='inner')
        action3 = action2[['Meal', 'Recipe ID']]
        action3 = action3.groupby('Meal')['Recipe ID'].apply(lambda x: x.iloc[np.random.randint(0, len(x))])

        # Output the optimal action at each episode so we can see how this changes over time
        actioninfull = np.append(actioninfull, action3)
        actioninfull = actioninfull.astype(int)

        # Rename for clarity
        SumofV = output
        OptimalActions = action3
        ActionsSelectedinTime = actioninfull

    return (SumofV, OptimalActions, ActionsSelectedinTime)


def get_recipe_values(result_model,meals):
    result = np.array(result_model[1])
    result_breakfast = meals.iloc[result[0] - 1, :]
    result_lunch = meals.iloc[10 + result[1] - 1, :]
    result_dinner = meals.iloc[20 + result[2] - 1, :]
    breakfast=result_breakfast[['Title', 'Price','Calories',"Carbs","Protein","Fat","Link"]]
    lunch=result_lunch[['Title', 'Price','Calories',"Carbs","Protein","Fat","Link"]]
    dinner=result_dinner[['Title', 'Price','Calories',"Carbs","Protein","Fat","Link"]]
    return breakfast,lunch,dinner


def nutrition_calculator(weight,height,aLevel,sex,age):
    if sex==1:
        BMR = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        BMR = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    activity_level={"1":1.2,"2":1.375,"3":1.55,"4":1.725,"5":1.9}
    tdee=BMR*activity_level[aLevel]/2
    protein=weight*1.1
    fat=weight*0.45
    carbs=(tdee-protein*4-fat*9)/4
    return tdee,protein,fat,carbs


'''                     Testing
calories,protein,fat,carbs=nutrition_calculator(70,170,"2",1,18)

print(calories,protein,fat,carbs)

Mdl1 =Model(data,alpha,num_episodes,epsilon,budget,reward)
print_values(Mdl1[1])

epsilon2=0;
epsilon3=1;
alpha2=0.1
Mdl2 =Model(data,alpha,num_episodes,epsilon2,budget,reward)
alpha3=0.01

Mdl3 =Model(data,alpha,num_episodes,epsilon3,budget,reward)
data_graph =  pandas.DataFrame({'Epsilon=0.5':Mdl1[0],'Epsilon=0':Mdl2[0],'Epsilon=1':Mdl3[0]})
print_values(Mdl3[1])

plt.figure()
data_graph.plot()
plt.title('Epsilon')
plt.xlabel('Episode')
plt.ylabel('Sum of V(a)')
plt.show()
print("Alpha")
Mdl1 =Model(data,alpha,num_episodes,epsilon,budget,reward)
print_values(Mdl1[1])
Mdl2 =Model(data,alpha3,num_episodes,epsilon,budget,reward)
print_values(Mdl2[1])
Mdl3 =Model(data,alpha2,num_episodes,epsilon,budget,reward)
print_values(Mdl3[1])
data_graph =  pandas.DataFrame({'Alpha=0.05':Mdl1[0],'Alpha=0.01':Mdl2[0],'Alpha=0.1':Mdl3[0]})
plt.figure()
data_graph.plot()
plt.title('Alpha')
plt.xlabel('Episode')
plt.ylabel('Sum of V(a)')
plt.show()
'''
