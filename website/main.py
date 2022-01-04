from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
import glob
import os
from werkzeug.utils import secure_filename


import classifier
import recommender as r

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super_secret!'
app.config["image_uploads"] = 'image-uploads/sample'
socketio = SocketIO(app)


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    files = glob.glob('./image-uploads/sample/*')
    for f in files:
        os.remove(f)

    os.makedirs("./image-uploads/sample", exist_ok=True)
    if request.method == "POST":
        # if request.files:
        images = request.files.getlist("image[]")

        for i in images:
            filename = secure_filename(i.filename)
            path = os.path.join(app.config["image_uploads"], filename)
            i.save(path)

        preds_list = classifier.predict_images("./image-uploads")

        print(preds_list)
        ingredients_list = [classifier.ingredient_dict[i] for i in preds_list]

        # return render_template("output_prediction.html", ingredients=ingredients_list)
        return process(ingredients_list)

    return render_template("uploadImage.html", ingr_list=classifier.ingredient_dict.values())


@app.route('/output_prediction/<i>')
def output_prediction(i):
    return render_template("output_prediction.html", ingredients=i)


@app.route('/rec')
def process(ingredients=classifier.ingredient_dict.values()):

    to_rate = r.recipes.sample(5)[["name", "recipe_id", "description"]]

    return render_template("see_results.html", ingredients=ingredients, recipes=to_rate.values.tolist())


@app.route("/recipes/<number>")
def serve_recipe(number):
    number = int(number)
    return render_template(
        "recipe.html",
        recipe_name=r.get_recipe_by_id(number, "name"),
        recipe_description=r.get_recipe_by_id(number, "description"),
        recipe_ingredients=r.get_recipe_by_id(number, "ingredients"),
        recipe_steps=r.get_recipe_by_id(number, "steps")
    )


@socketio.on("get_recommendations")
def create_recommendations(json):
    print(json)
    ingr = json["ingredients"]
    method = json["method"]
    prefs = {i['name']: i['value'] for i in json["form"]}
    print(prefs)
    if method == "retrain":
        results = r.deep_recommend(
            prefs, ingr, progress_func=lambda x: emit("progress", x))
    else:
        results = r.RF_recommend(
            prefs, ingr, progress_func=lambda x: emit("progress", x))
    emit("recommendations", [{"name": r.get_recipe_by_id(
        i[0], "name"), "id":int(i[0]), "est_rating": i[1]} for i in results[:20]])


if __name__ == "__main__":
    socketio.run(app, debug=True)
