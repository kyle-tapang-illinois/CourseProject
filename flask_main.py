from flask import Flask, render_template, request, redirect, url_for, render_template
from Ranker import rankBySentiment
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

	#results = rankBySentiment(query)
	#print(results.head(5))
	df = pd.DataFrame({"title": ['Hacker News Parody Thread', 
	                             'Kernel (OS Kernel Book)',
								 '10 KB Club: Curated list of websites whose home pages do not exceed 10 KB size',
								 'Eye contact marks the rise and fall of shared attention in conversation',
								 'Mildly Interesting Quirks of C'],
                     
                   "url": ['http://bradconte.com/files/misc/HackerNewsParodyThread/',
				           'https://539kernel.com/',
						   'https://10kbclub.com/',
						   'https://www.pnas.org/doi/10.1073/pnas.2106645118',
						   'https://gist.github.com/fay59/5ccbe684e6e56a7df8815c3486568f01'],
                     
                    })
	titles = df["title"]
	urls = df["url"]

	return render_template("results.html", titles=titles, urls=urls)
	

if __name__ == "__main__":
	app.run(debug=True)


