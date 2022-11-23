from flask import Flask, render_template, request, redirect, url_for, render_template
from Ranker import ranker
import pandas as pd

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def home():
	error = None

	if request.method == "POST":
		query= request.form["nm"]
		if query !="":
			return redirect(url_for("results", query=query))
		else:
			error = 'Please enter your query'
			return render_template("home.html", error = error)
	else:
		return render_template("home.html")

@app.route("/results/<string:query>" , methods=["GET"])
def results(query):
	df = ranker(query)
	titles = df["title"].values
	urls = df["url"].values

	return render_template("results.html", titles=titles, urls=urls)
	

if __name__ == "__main__":
	app.run(debug=True)


