import collections
import pickle
import networkx as nx
from numpy import random
import numpy as np
import matplotlib.pylab as plt
from matplotlib import gridspec
import pandas as pd
import csv
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from tabulate import tabulate

def countHashtags(hashtags):
    '''
    hashtags = panda dataframe with a column 'Hashtags'
    returns a dictionary of hastags and their counts

    '''
    final_split = []
    splitted = []
    for hashtag in hashtags['Hashtags']:
        splitted = hashtag.split(", ")
        #print(splitted)
        for i in splitted:
            i = i.replace("'", "")
            i = i.replace("{", "")
            i = i.replace("}", "")
            final_split.append(i)
    
    counted = Counter(final_split)

    print(final_split)

    return counted, final_split
        
def drawHistogram(data):
    '''
    Draws a histogram 
    data = dictionry of hashtags and their counts
    '''
    plt.hist(data.values())
    plt.show()

def drawPiechart(data):
    '''
    Draws a piechart
    data = pandas dataframe with a column 'Language'
    '''
    languages = data['Language'].to_numpy()
    counted = Counter(languages)
    plt.pie(counted.values(), labels=counted.items())
    plt.show()

def textAnalyze(sentences):
    '''
    Sentences should be an array of strings.
    Uses VADER for analyzing strings.
    Returns an array with compound scores between [-1, 1]
    Where...
    positive sentiment: compound score >= 0.05
    neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
    negative sentiment: compound score <= -0.05
    '''
    analyzer = SentimentIntensityAnalyzer()
    analyzed = []
    for sentence in sentences:
        analyzed.append(analyzer.polarity_scores(sentence))
    
    return analyzed

def ternaryplot(data):
    fig = px.scatter_ternary(data, a="pos", b="neu", c="neg")
    fig.show()

def makeGraph(tweets):
    '''
    Makes a graph where nodes are hashtags and edges between
    nodes indicate there is atleast one post with that hashtag.
    tweets = dictionary of tweets
    '''
    A = np.array(tweets)
    A = nx.from_numpy_array(A, create_using=nx.MultiGraph)
    A.edges(data=True)
    nx.draw(G)

    plt.show()

def calculateProperties(G):
    '''
    Makes a table of global properties of the graph.
    Takes the graph constructed in makegraph() as input.
    Number of nodes, number of edges, average degree centrality, diameter, clustering coefficient, size of largest component
    '''

    
    nodes = G.number_of_nodes() #count number of nodes in graph
    edges = G.number_of_edges() #count number of edges in graph
    degreeCent = nx.degree_centrality(G) #Calculate degree centrality of graph
    diameter = nx.diameter(G, e=None, usebounds=False) #diameter of graph
    cluster = nx.clustering(G, nodes=None, weight=None)#clustering coefficient of nodes
    largestComponent = max(nx.connected_component_subgraphs(G), key=len) #Largest component of graph.

    data = [
        ["nodes", "nodes"],
        ["edges", "edges"],
        ["Degree centrality", "degreeCent"],
        ["diameter", "diameter"], 
        ["Cluster", "cluster"]
        ["Largest component", "largestComponent"]
    ]

    head = ["Name", "Value"]
    print(tabulate(data, headers=head, tablefmt="grid"))













    





if __name__ == "__main__":
    #Transform csv data to dictionary
    #with open('tweetsdata_00.csv',newline='', encoding="utf8") as csvfile:
    #    reader = csv.reader(csvfile)
     #   next(reader)
        # pull in each row as a key-value pair
    #    dictionary_of_tweets = dict(reader)
    
    df = pd.read_csv('tweetsdata_00.csv', encoding="utf8", usecols=['Language','Text','Hashtags'])
    hashtags = df['Hashtags'].to_numpy()
    
    #print(hashtags)
    
    #Draw a histogram showing the popularity of the main hashtags highlighting the number of posts per individual hashtag.
    counted_dict, final = countHashtags(df)
    #drawHistogram(counted_dict)
    #Use a pie chart illustrations to show the language of the posts for each of the above main hashtags.
    #drawPiechart(df)

    #analyzedArray = textAnalyze(df['Text'].to_numpy())
    #print(analyzedArray)
    #ternaryplot(analyzedArray)
    

    '''
    #Use VADER tool (https://github.com/cjhutto/vaderSentiment), which output sentiment in terms
    #of POSITIVE, NEGATIVE and NEUTRAL to determine the sentiment of each post of the dataset.
    #Then represent the sentiment of each tweet as a point in the ternary plot.

    analyzedArray = textAnalyze(df['Text'].to_numpy())

    ternaryplot(analyzedArray)
    #build a social graph where each node corresponds to a hashtag and an edge
    #between hashtag A and hashtag B indicates that there is at least one post which contains both
    #hashtag A and hashtag B
    '''
