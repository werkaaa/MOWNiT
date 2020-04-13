from flask import Flask, render_template, request
from wtforms import Form, validators, StringField, IntegerField
from search_no_idf import SearchNoIDF
from search_svd import SearchSVD
# App config.
DEBUG = False
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

texts = None

class ReusableForm(Form):
    query = StringField('Name:', validators=[validators.required()])
    k = IntegerField('Name:', validators=[validators.required()])

@app.route("/", methods=['GET', 'POST'])
def start():
    form = ReusableForm(request.form)
    return render_template('start.html', form=form)

@app.route("/search_no_idf", methods=['GET', 'POST'])
def search_no_idf():
    s = SearchNoIDF()
    s.load()
    global texts
    texts = s.texts

    form = ReusableForm(request.form)
    result = ""
    query = ""
    print(form.errors)
    if request.method == 'POST':
        query = request.form['query']
        k = request.form['k']
        if not k:
            k = 1
        query_ans = s.search(query, int(k))
        if query_ans == 'No similarity found':
            result = []
        else:
            result = [[id, s.texts[id], round(val, 3)] for (id, val) in query_ans]

    return render_template('hello.html', form=form, output=result, query=query)

@app.route("/search_raw", methods=['GET', 'POST'])
def search_raw():
    s = SearchSVD()
    s.load()
    global texts
    texts = s.texts

    form = ReusableForm(request.form)
    result = ""
    query = ""
    print(form.errors)
    if request.method == 'POST':
        query = request.form['query']
        k = request.form['k']
        if not k:
            k = 1
        query_ans = s.search(query, int(k))
        if query_ans == 'No similarity found':
            result = []
        else:
            result = [[id, s.texts[id], round(val, 3)] for (id, val) in query_ans]

    return render_template('hello.html', form=form, output=result, query=query)

@app.route("/search_svd", methods=['GET', 'POST'])
def search_svd():
    s = SearchSVD()
    s.load()
    s.load_svd()
    global texts
    texts = s.texts

    form = ReusableForm(request.form)
    result = ""
    query = ""
    print(form.errors)
    if request.method == 'POST':
        query = request.form['query']
        k = request.form['k']
        if not k:
            k = 1
        query_ans = s.search_svd(query, int(k))
        if query_ans == 'No similarity found':
            result = []
        else:
            result = [[id, s.texts[id], round(val, 3)] for (id, val) in query_ans]

    return render_template('hello.html', form=form, output=result, query=query)

@app.route("/result", methods=['GET', 'POST'])
def result():

    output = ""
    if request.method == 'POST':
        file_num = request.form['index']
        f = open(f'wiki/files/{file_num}.txt', "r")
        output = f.read()
        f.close()


    # if form.validate():
    #         # Save the comment here.
    #     flash('Hello ' + query)
    # else:
    #     flash('All the form fields are required. ')

    return output

if __name__ == "__main__":
    app.run()
