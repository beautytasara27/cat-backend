import csv
import json

import catsim.plot as catplot
import matplotlib.pyplot as plt
from catsim.irt import icc
from catsim.stopping import *
from flask import Flask, request
from flask_cors import CORS

from Item import Item
from cat import Irt

app = Flask(__name__)
CORS(app)


@app.route('/start', methods=['GET'])
def start():
    if not request.is_json:
        return "Content type is not supported."
    global item_bank
    global item_difficulties
    global items
    global responses
    global est_theta
    global responses
    global thetas
    global administered_items
    global ad_items
    global model
    global next_item_index

    item_bank, item_difficulties = read_csv('irt_dataset.csv')
    items = numpy.zeros((len(item_difficulties), 4))
    items[:, 1] = item_difficulties
    items[:, 0] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # use catsim for normalization
    items[:, 3] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    model = Irt(items)
    responses = []
    est_theta = model.estimate_theta()
    thetas = [est_theta]
    next_item_index = model.next_item(est_theta)
    administered_items = []
    ad_items = []  # in order of administration

    return {

        "itemCode": item_bank[next_item_index].itemCode,
        "question": item_bank[next_item_index].question,
        "options": item_bank[next_item_index].options
    }


@app.route('/next', methods=['POST'])
def next_item():
    global est_theta
    global next_item_index

    # _stop = model.stopper.stop(administered_items=items[administered_items], theta=est_theta)
    if request.is_json:

        data = request.json
        score = score_question(data.get('response').lower(), item_bank[next_item_index].answer.lower())
        responses.append(score)
        thetas.append(est_theta)
        administered_items.append(next_item_index)
        ad_items.append(items[next_item_index].tolist())
        est_theta = model.estimate_theta(administered_items, responses, est_theta)
        next_item_index = model.next_item(est_theta, administered_items)
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
