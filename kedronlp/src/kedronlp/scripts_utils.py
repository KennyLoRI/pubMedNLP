import ast
import csv
import sys
import numpy as np


# needed for row_preprocessing
def get_descriptors(descriptor_string):
    descriptor_list = []
    tmp = descriptor_string.replace("\'", "").split("StringElement(")
    for item in tmp:
        pos = item.find(",")
        if pos != -1:
            descriptor = item[:pos]
            descriptor_list.append(descriptor)
    return descriptor_list

def preprocess_row(row):
    for key in row.keys():
        if type(row[key]) != str:
            if np.isnan(row[key]):
                row[key] = "NA"
            else:
                row[key] = str(row[key])
    for key in ["Authors", "Affiliations", "Qualifier", "Major Qualifier"]:
        if row[key] != "NA":
            row[key] = ast.literal_eval(row[key])
        if key == "Authors":
            author_list = []
            for author in row[key]:
                if "," in author:
                    lastname, firstname = author.split(", ")
                    fullname = firstname + " " + lastname
                else:
                    fullname = author
                author_list.append(fullname)
            row[key] = author_list
        else:
            row[key] = []
    for key in ["Descriptor", "Major Descriptor"]:
        if row[key] != "NA":
            row[key] = get_descriptors(row[key])
        else:
            row[key] = []
    return row

def get_combined_doc(row):
    combined_doc = ""
    for key, value in row.items():
        if not value:
            value = "NA"
        elif type(value) == list:
            value = ", ".join(value)
        combined_doc += key + ": " + value + "\n"
    return combined_doc

def get_doc_info(row):
    doc_info = ""
    for key, value in row.items():
        if key != "Abstract":
            if not value:
                value = "NA"
            elif type(value) == list:
                value = ", ".join(value)
            doc_info += key + ": " + value + "\n"
    return doc_info

def increase_csv_maxsize():
    maxInt = sys.maxsize

    while True:
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    csv.field_size_limit(sys.maxsize)