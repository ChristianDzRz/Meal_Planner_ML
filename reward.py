import pandas
import numpy as np
import math
breakfast = pandas.read_csv('breakfast.csv', encoding='utf-8', delimiter=";")
lunch=pandas.read_csv('lunch.csv', encoding='utf-8', delimiter=";")
dinner=pandas.read_csv('dinner.csv', encoding='utf-8', delimiter=";")
meals=pandas.concat([breakfast,lunch,dinner])
allergens_list=["eggs"]
preference_list=["eggs"]
recipes=np.array(meals["Ingredients"])
reward=np.zeros(len(recipes))
index=0
print(breakfast.iloc[1, 0])
for recipe in recipes:
    no_ingredients = 0
    for allergen in allergens_list:
        for ingredient in eval(recipe):
            if(allergen in ingredient):
                reward[index]+=-10
    for preference in preference_list:
        for ingredient in eval(recipe):
            if(preference in ingredient):
                no_ingredients+=1
                reward[index]+=math.log(3*no_ingredients+1)
    index+=1
