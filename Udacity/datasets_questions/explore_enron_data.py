#!/usr/bin/python
# -*- coding: utf-8 -*-

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
#print enron_data
#print len(enron_data) #146
#print len(enron_data[enron_data.keys()[0]]) #21
n = [x for x in enron_data.values() if x['poi'] ==True ]
#print(len(n)) #18
poi_name_record = open("../final_project/poi_names.txt").read().split("\n")
poi_name_total = [record for record in poi_name_record if "(y)" in record or "(n)" in record]
#printlen(poi_name_total)
"""
Like any dict of dicts, individual people/features can be accessed like so:

enron_data["LASTNAME FIRSTNAME"]["feature_name"]
or, sometimes 
enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"]["feature_name"]
"""
#What is the total value of the stock belonging to James Prentice?
#print enron_data["PRENTICE JAMES"]["total_stock_value"] #1095040
#How many email messages do we have from Wesley Colwell to persons of interest?
#print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"] #11
#What’s the value of stock options exercised by Jeffrey K Skilling?
#print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"] #19250000

#(Lay, Skilling and Fastow), max total_payments, and what was that value?
#print max(enron_data["LAY KENNETH L"]["total_payments"],enron_data["SKILLING JEFFREY K"]["total_payments"],enron_data["FASTOW ANDREW S"]["total_payments"])
"""
execs = [s for s in enron_data.keys() if ("SKILLING" in s) or ("LAY" in s) or ("FASTOW" in s) ] 
max( [(enron_data[person]['total_payments'],person) for person in execs] ) 
"""
#quiz 27
"""
len([enron_data[person]['salary'] for person in enron_data if enron_data[person]['salary'] != 'NaN']) # 95
len([enron_data[person]['email_address'] for person in enron_data if enron_data[person]['email_address'] != 'NaN' ] # 111
"""