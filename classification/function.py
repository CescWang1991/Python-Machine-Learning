import numpy as np
from numpy import *
import pandas as pd


def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    fr = open(filename)
    numberOfColumns = len(fr.readline().split(' ')) - 2
    labels = zeros((numberOfLines, 1))
    features = zeros((numberOfLines, numberOfColumns))
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(' ')
        labels[index, 0] = listFromLine[0]
        for vector in listFromLine[1:len(listFromLine)]:
            list = vector.split(":")
            features[index, int(list[0])-1] = list[1]
        index += 1
    return labels, features


def libsvm2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    labels = zeros((numberOfLines, 1))
    features = []
    fr = open(filename)
    numberOfColumns = 0
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(' ')
        labels[index, 0] = listFromLine[0]
        index += 1
        featureLine = []
        firstFeatureIndex, firstFeature = listFromLine[1].split(':')
        lastFeatureIndex, lastFeature = listFromLine[1].split(':')
        print(firstFeatureIndex)
        for vector in listFromLine[1:len(listFromLine)]:
            list = vector.split(":")
            print(list[0])
            featureLine[int(list[0]) - 1] = list[1]
            features = np.append(features, featureLine)
    return features


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]
