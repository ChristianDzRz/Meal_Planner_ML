from flask import Flask, render_template, request
from flask_assets import Environment
from model import run_model
app = Flask(__name__)
assets = Environment(app)

@app.route('/', methods=["GET","POST"])
def menu():
    value=0
    if request.method =="POST":
        req = request.form
        age= req.get('age')
        height= req.get('height')
        weight= req.get('weight')
        budget= req.get('budget')
        allergies=req.get('allergies')
        ingredients=req.get('ingredients')
        sex=req.get('sex')
        activity=req.get('activity')
        breakfast,lunch,dinner=run_model(int(age),int(weight),float(height),
                                         int(budget),ingredients,allergies,activity,sex)
        value=1
    if value==1:
        return render_template('menu.html',breakfast_title = breakfast['Title'], price_b = breakfast['Price'], fat_b = breakfast['Fat'],
        carbs_b = breakfast['Carbs'], protein_b = breakfast['Protein'], calories_b=breakfast["Calories"],
        lunch_title = lunch['Title'], price_l = lunch['Price'], fat_l = lunch['Fat'],
        carbs_l = lunch['Carbs'], protein_l = lunch['Protein'], calories_l=lunch["Calories"],
        dinner_title = dinner['Title'], price_d = dinner['Price'], fat_d = dinner['Fat'],
        carbs_d = dinner['Carbs'], protein_d = dinner['Protein'], calories_d=dinner["Calories"],
        )
    else:
        return render_template('menu.html')

if __name__ == "__main__":
    app.run()
