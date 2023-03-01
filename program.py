"""This file contains code for running the CAT algorithm using Catsim Package
   : It runs on Flask
"""
import csv

from catsim.irt import normalize_item_bank
from catsim.stopping import *
from flask import Flask, request, session
from flask_cors import CORS, cross_origin
from flask import json
from Item import Item
from cat import Irt
from flask_session import Session
from werkzeug.exceptions import HTTPException
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'your secret key'
app.config["SESSION_PERMANENT"] = False
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
"""
The session is used to store user specific variables during the test which will be deleted upon completion.
"""

"""
This is a generic handler for HTTP errors, it returns readable errors in json format
"""
@app.errorhandler(HTTPException)
def handle_exception(e):
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response


"""
    The endpoint for starting the Test, 
    : It accepts json type content, and the body contains the identifier of the participant e.g.
    : {
        "id":"some string"
      }
    : The id is used as the key to the session object initialized earlier,
"""


@app.route('/api/v1/start', methods=['GET'])
@cross_origin(supports_credentials=True)
def start():
    if not request.is_json:
        return "Content type is not supported."
    key = session.get('key')
    if key is None:
        session['key'] = request.args.get("id")
        key = session.get('key')
    item_bank, item_difficulties = read_csv('irt_dataset.csv')
    items = numpy.zeros((len(item_difficulties), 1))
    items[:, 0] = item_difficulties
    items = normalize_item_bank(items)
    model = Irt(items)
    responses = []
    est_theta = model.estimate_theta()
    thetas = [est_theta]
    next_item_index = model.next_item(est_theta)
    administered_items = []
    # user specific values are stored in the session so that they can be accessed from a different endpoint.
    session["data"] = [model, responses, administered_items, next_item_index, est_theta, thetas, item_bank]
    # the first question returned to the user
    return {

        "itemCode": item_bank[next_item_index].itemCode,
        "question": item_bank[next_item_index].question,
        "options": item_bank[next_item_index].options
    }


"""
    This is the endpoint for requesting the next question in the test.
    : It accepts json content type
    : The body contains the response to the previous question e.g.
    {
        "response": "A"
    }
"""


@app.route('/api/v1/next', methods=['POST'])
@cross_origin(supports_credentials=True)
def next_item():
    if request.is_json:
        key = session.get('key')
        # get the previously stored user specific variables
        model, responses, administered_items, next_item_index, est_theta, thetas, item_bank = session["data"]
        data = request.json
        # score the user response and add it to the list of responses
        score = score_question(data.get('response').lower(), item_bank[next_item_index].answer.lower())
        responses.append(score)
        # also keep track of administered questions and estimated abilities so far
        thetas.append(est_theta)
        administered_items.append(next_item_index)
        # calculate new estimated abilities based on the latest score
        est_theta = model.estimate_theta(administered_items, responses, est_theta)
        # calculate the next question based on the user's current abilities
        next_item_index = model.next_item(est_theta, administered_items)
        # update the stored values
        session["data"] = [model, responses, administered_items, next_item_index, est_theta, thetas, item_bank]
        # criteria for ending the text, after all the items are completed or after a set number of questions
        if next_item_index is None or len(administered_items) == 10:
            # calculate the final proficiency of the participant
            proficiency = estimate_proficiency(est_theta, administered_items)
            write_csv([key, *responses])
            return {
                "complete": 1,
                "message": "Test Completed!",
                "theta": est_theta,
                "responses": responses,
                "proficiency": proficiency
            }
        # if there are still other questions to be administered return this
        return {"complete": 0,
                "data": {
                    "itemCode": item_bank[next_item_index].itemCode,
                    "question": item_bank[next_item_index].question,
                    "options": item_bank[next_item_index].options
                }
                }
    else:
        return "Content type is not supported."


"""
    Read MCQ  questions from the CSV file,
    : The file must have the following headings in order
        itemCode : unique identifier for the question,
        difficulty : any values, they will be normalized to the range (-4 < x < 4),
        item : The question,
        key : The answer,
        (After these fields the remainders are possible options of the MCQ, doesn't have to be 4)
        A : "value",
        B : "value",
        C : "value",
        etc
        
"""


def read_csv(csv_name):
    filename = open(csv_name)
    file = csv.reader(filename)
    next(file, None)
    item_bank = []
    item_difficulties = []
    for row in file:
        item = Item(*row[0:4], [*row[4:]])
        item_bank.append(item)
        item_difficulties.append(row[1])
    return item_bank, normalize(numpy.array(item_difficulties).astype(numpy.float))


"""
    This is the scoring function, for a correct solution, return True else False
"""


def score_question(response, answer):
    if response == answer:
        s = True
    else:
        s = False
    return s


"""
    This is the function that estimates a participant's final ability,
    : The ability is in the range (-4 < x < 4)
    : The boundaries can be any values, the higher the ability, the better the examinee's proficiency
"""


def estimate_proficiency(est_theta, administered_items, lower=1, upper=2):
    assert administered_items is not None
    if est_theta < lower:
        return "Novice"
    elif lower < est_theta < upper:
        return "Proficient"
    elif est_theta > upper:
        return "Expert"


def normalize(x):
    x_min, x_max = numpy.min(x), numpy.max(x)
    norm = (x - x_min) / (x_max - x_min)
    new_range = (-4, 4)
    return norm * (new_range[1] - new_range[0]) + new_range[0]


"""
    This file keeps track of all the users who have taken the test
"""


def write_csv(row):
    with open('output.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)


if __name__ == '__main__':
    app.run()
