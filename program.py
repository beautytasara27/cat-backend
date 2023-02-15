import csv
import json

import catsim.plot as catplot
import matplotlib.pyplot as plt
from catsim.irt import icc, normalize_item_bank
from catsim.stopping import *
from flask import Flask, request
from flask_cors import CORS

from Item import Item
from cat import Irt
from flask import session

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'your secret key'

dict = {}
@app.route('/start', methods=['POST'])
def start():
    if not request.is_json:
        return "Content type is not supported."
    key = session.get('key')
    if key is None:
        data = request.json
        session['key'] = data.get("id")

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
    dict[key] = [model, responses, administered_items, next_item_index, est_theta, thetas, item_bank]
    return {

        "itemCode": item_bank[next_item_index].itemCode,
        "question": item_bank[next_item_index].question,
        "options": item_bank[next_item_index].options
    }


@app.route('/next', methods=['POST'])
def next_item():
    # _stop = model.stopper.stop(administered_items=items[administered_items], theta=est_theta)
    if request.is_json:
        key = session.get('key')
        model, responses, administered_items, next_item_index, est_theta, thetas, item_bank = dict[key]
        data = request.json
        score = score_question(data.get('response').lower(), item_bank[next_item_index].answer.lower())
        responses.append(score)
        thetas.append(est_theta)
        administered_items.append(next_item_index)
        est_theta = model.estimate_theta(administered_items, responses, est_theta)
        next_item_index = model.next_item(est_theta, administered_items)
        dict[key] = [model, responses, administered_items, next_item_index, est_theta, thetas, item_bank]
        if next_item_index is None:
            return json.dumps(
                {
                    "complete": 1,
                    "message": "Test Completed!",
                    "theta": est_theta,
                    "responses" : responses
                }
            )
        return json.dumps({"complete": 0,
                           "data": {
                               "itemCode": item_bank[next_item_index].itemCode,
                               "question": item_bank[next_item_index].question,
                               "options": item_bank[next_item_index].options
                           }
                           })
    else:
        return "Content type is not supported."


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
    return item_bank, item_difficulties


def score_question(response, answer):
    if response == answer:
        s = True
    else:
        s = False
    return s


def write_csv(row):
    with open('output.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    item_bank, item_difficulties = read_csv('irt_dataset.csv')
    # print(item_difficulties)
    items = numpy.zeros((len(item_difficulties), 4))
    items[:, 1] = item_difficulties
    items[:, 0] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # use catsim for normalization
    items[:, 3] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    model = Irt(items)
    a, b, c, d = items[0]
    catplot.item_curve(a, b, c, d)
    plt.bar([x for x in range(0, 10)], item_difficulties)
    plt.show()
    responses = []
    est_theta = model.estimate_theta()
    thetas = [est_theta]
    next_item_index = model.next_item(est_theta)
    administered_items = []
    ad_items = []
    print('Answer the questions to the best of your ability')
    proceed = input("Proceed ? y / n ")
    _stop = False
    if proceed:
        while not _stop:
            print()
            print(item_bank[next_item_index].itemCode + "." + item_bank[next_item_index].question)
            print()
            choices = ['A', 'B', 'C', 'D']
            for index in range(0, 4):
                print(choices[index] + ". " + item_bank[next_item_index].options[index])
            response = input()
            if response.upper() not in choices:
                response = input("Please respond with the correct letter")
            print("Answer: ", item_bank[next_item_index].answer)
            score = score_question(response.lower(), item_bank[next_item_index].answer.lower())
            responses.append(score)
            administered_items.append(next_item_index)
            ad_items.append(items[next_item_index].tolist())
            est_theta = model.estimate_theta(administered_items, responses, est_theta)
            thetas.append(est_theta)
            print("theta: ", est_theta)
            next_item_index = model.next_item(est_theta, administered_items)
            if next_item_index is not None:
                # true_theta = 0.8
                a, b, c, d = items[next_item_index]
                prob = icc(est_theta, a, b, c, d)

                print('Probability to correctly answer item:', prob)
                print('Did the user answer the selected item correctly?', score)

        write_csv(responses)
        print(thetas)
        print(numpy.asarray(ad_items))
        print(items)
        catplot.test_progress(thetas=thetas, administered_items=numpy.asarray(ad_items), true_theta=0.8)


if __name__ == '__main__':
    app.run()
