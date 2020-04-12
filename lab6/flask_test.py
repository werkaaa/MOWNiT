from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField, IntegerField
from search import Search
from search_normalized import SearchNormalized
# App config.
DEBUG = True
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

@app.route("/search", methods=['GET', 'POST'])
def search():
    s = Search()
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
            result = [[id, s.texts[id]] for id in query_ans]

    return render_template('hello.html', form=form, output=result, query=query)

@app.route("/search_normalized", methods=['GET', 'POST'])
def search_normalized():
    s = SearchNormalized()
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
            result = [[id, s.texts[id]] for id in query_ans]

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
