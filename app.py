from flask import Flask, render_template, request, redirect, url_for, render_template
from Ranker import ranker
import pandas as pd

app = Flask(__name__)


@app.route("/", methods=["GET","POST"])
def home():
	error = None

	if request.method == "POST":
		searchData= request.form["searchData"]
		if searchData !="":
			return redirect(url_for("hackerNewsSearchResults", searchData=searchData))
		else:
			error = 'Please enter your search data'
			return render_template("home.html", error = error)
	else:
		return render_template("home.html")

@app.route("/hackerNewsSearchResults/<string:searchData>" , methods=["GET"])
def hackerNewsSearchResults(searchData):
	df = ranker(searchData)
	titles = df["title"].values
	urls = df["link"].values

	return render_template("searchResults.html", titles=titles, urls=urls)
	

if __name__ == "__main__":
	app.run()


