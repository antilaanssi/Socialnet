from ast import increment_lineno, parse
import collections
from email.utils import parsedate
import pickle
from xml.dom.minicompat import NodeList
from xml.etree.ElementInclude import include
import networkx as nx
from matplotlib.cm import ScalarMappable
from numpy import random
import numpy as np
import matplotlib.pylab as plt
from matplotlib import gridspec
import pandas as pd
import csv
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
from networkx.algorithms import community
import scipy
from scipy import stats

class tweet():

    name = ""
    occurrences = 0
    totalLikes = 0
    totalRetweets = 0
    totalReplies = 0
    totalQuotes = 0

    def __init__(self, name):
        self.name = name
    
    def AddLikes(self, retweets, reply, likes, quote):
        self.occurrences +=1
        self.totalLikes += int(likes)
        self.totalRetweets += int(retweets)
        self.totalReplies += int(reply)
        self.totalQuotes += int(quote)

    def PrintInfo(self):
        #do nothing
        #print("NAME: " + self.name)
        #print("Occurances: " + str(self.occurrences))
        #print("totalLikes: " + str(self.totalLikes))
        #print("totalRetweets: " + str(self.totalRetweets))
        #print("totalReplies: " + str(self.totalReplies))
        #print("totalQuotes: " + str(self.totalQuotes))
        #print("\n")
        a=a

#"{'retweet_count': 34, 'reply_count': 0, 'like_count': 0, 'quote_count': 0}"

def NodeNameExists(Nodes, target):
    exists = False
    for currentTweet in Nodes:
        if currentTweet.name == target:
            exists = True
            break
    return exists

def NodeNameMatches(Node, target):
    return Node == target



def countHashtags(hashtags):
    '''
    hashtags = panda dataframe with a column 'Hashtags'
    returns a dictionary of hastags and their counts

    '''
    final_split = []
    splitted = []
    for hashtag in hashtags['Hashtags']:
        print(hashtag)
        if (pd.isna(hashtag) == True):
            continue
        splitted = hashtag.split(", ")
        for i in splitted:
            i = i.replace("'", "")
            i = i.replace("{", "")
            i = i.replace("}", "")
            i = i.replace(".", "")
            final_split.append(i)
    
    counted = Counter(final_split)

    return counted, final_split
            


def parseTweets(tweetsRaw):
    parsedHashtags = []
    TweetClasses = []
    for hashtag in tweetsRaw['Hashtags']:
        if (pd.isna(hashtag) == True):
            continue
        tempList = []
        splitted = hashtag.split(", ")
        for i in splitted:
            i = i.replace("'", "")
            i = i.replace("{", "")
            i = i.replace("}", "")
            tempList.append(i)

            if not NodeNameExists(TweetClasses, i):
                TweetClasses.append(tweet(i))

        parsedHashtags.append(tempList)
    
    return parsedHashtags, TweetClasses


def ParsePublicMetrics(tweetsRaw):
    parsedSocials = []
    for hashtag in tweetsRaw['Public_metrics']:
        if (pd.isna(hashtag) == True):
            continue
        tempList = []
        splitted = hashtag.split(", ")
        for i in splitted:
            i = i.replace("'", "")
            i = i.replace("{", "")
            i = i.replace("}", "")
            i = i.replace("retweet_count", "")
            i = i.replace("reply_count", "")
            i = i.replace("like_count", "")
            i = i.replace("quote_count", "")
            i = i.replace(":", "")
            tempList.append(i)

        parsedSocials.append(tempList)
    
    return parsedSocials

        
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
        if (pd.isna(sentence) == True):
            continue
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
    
    plt.show()


def calculateProperties(G):
    '''
    Makes a table of global properties of the graph.
    Takes the graph constructed in makegraph() as input.
    Number of nodes, number of edges, average degree centrality, diameter, clustering coefficient, size of largest component
    '''
 
    nodes = G.number_of_nodes() #count number of nodes in graph
    edges = G.number_of_edges() #count number of edges in graph
    degreeCent = calculateDegreeCentrality(G, nodes)
    roundedDegreeCent = round(degreeCent, 4)
    #diameter = nx.diameter(G, e=None, usebounds=False) #diameter of graph
    
    
    cluster = calculateCluster(G, nodes)#clustering coefficient of nodes
    roundedCluster = round(cluster, 4)

    #Largest component of graph.
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])

    #remove parentheses from right side to have the actual value in the table.
    data = [
        ["Nodes", nodes],
        ["Edges", edges],
        ["Degree centrality", roundedDegreeCent],
        ["Diameter", "Infinite because disconnected"], 
        ["Cluster", roundedCluster],
        ["Largest component", G0]
    ]

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    #hide axles
    ax.axis('off')
    ax.axis('tight')

    #show table
    table = ax.table(cellText=data, loc='center')
    fig.tight_layout()
    plt.show()

def calculateDegreeCentrality(G, nodes):

    '''
    Calculates average degree centrality of whole graph.
    '''

    degreeCent = nx.degree_centrality(G) #Calculate degree centrality of graph
    


    new_output = sum(degreeCent.values())
    result = new_output/nodes

    return result

def calculateCluster(G, total_nodes):

    '''
    Calculates average clustering coefficient of whole graph.
    '''
    
    cluster = nx.clustering(G, nodes=None, weight=None) #Calculate cluster coefficient of graph


    new_output = sum(cluster.values())
    result = new_output/total_nodes

    return result


    

def plotDegree(G):
    '''
    Plot degree distribution of graph
    '''


    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees)
    plt.show()

def plotLocal(G):

    gc = G.subgraph(max(nx.connected_components(G)))
    lcc = nx.clustering(gc)

    fig, (ax2) = plt.subplots(ncols=1, figsize=(12, 4))

    ax2.hist(lcc.values(), bins=10)
    ax2.set_xlabel('Clustering')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def labelPropagation(G):


    communities = community.label_propagation_communities(G)

    set_node_community(G, communities)
    set_edge_community(G)
    node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]
    # Set community color for edges between members of the same community (internal) and intra-community edges (external)
    external = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] == 0]
    internal = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] > 0]
    internal_color = ['black' for e in internal]
    

    pos = nx.spring_layout(G)
    plt.rcParams.update({'figure.figsize': (15, 10)})
    # Draw external edges
    nx.draw_networkx(
        G,
        pos=pos,
        node_size=0,
        edgelist=external,
        edge_color="silver")
    # Draw nodes and internal edges
    nx.draw_networkx(
        G,
        pos=pos,
        node_color=node_color,
        edgelist=internal,
        edge_color=internal_color)

    plt.show()

def set_node_community(G, communities):
        '''Add community to node attributes'''
        for c, v_c in enumerate(communities):
            for v in v_c:
                # Add 1 to save 0 for external edges
                G.nodes[v]['community'] = c + 1

def set_edge_community(G):
        '''Find internal edges and add their community to their attributes'''
        for v, w, in G.edges:
            if G.nodes[v]['community'] == G.nodes[w]['community']:
                # Internal edge, mark with community
                G.edges[v, w]['community'] = G.nodes[v]['community']
            else:
                # External edge, mark as 0
                G.edges[v, w]['community'] = 0

def get_color(i, r_off=1, g_off=1, b_off=1):
        '''Assign a color to a vertex.'''
        r0, g0, b0 = 0, 0, 0
        n = 16
        low, high = 0.1, 0.9
        span = high - low
        r = low + span * (((i + r_off) * 3) % n) / (n - 1)
        g = low + span * (((i + g_off) * 5) % n) / (n - 1)
        b = low + span * (((i + b_off) * 7) % n) / (n - 1)
        return (r, g, b)

def build_hashtag_graph(list_of_hashtags, parsedTweets):
    G = nx.Graph()
    G.add_nodes_from(list_of_hashtags)

    for allHashtagsInTweet in parsedTweets: #lista hashtageista esim. [Ukraine, Russia]
        for hashtags in allHashtagsInTweet: # yksittäiset hashtagit esim. Ukraine -> Russia
            current = hashtags
            for hashtags in allHashtagsInTweet: # yksittäiset hashtagit, mutta current on yksitellen Esim. Current: Ukraine, Ukraine -> Russia. Current: Russia, Ukraine -> Russia
                if current != hashtags:
                    G.add_edge(current, hashtags)

    nx.draw(G, with_labels = True)
    plt.show()
    
    return G


def FetchSocialAttributes(socials, parsedHashtags, parsedSocialInfoClasses):
    currentTweetIndex = 0
    for allHashtagsInTweet in parsedHashtags: #lista twiittien hashtageista
        for individualHashtags in allHashtagsInTweet: #yksittäisiä hashtageja
            currentNode = 0
            for hashtagClasses in parsedSocialInfoClasses:
                if (currentNode > len(parsedSocialInfoClasses)):
                    continue
                if NodeNameMatches(parsedSocialInfoClasses[currentNode].name, individualHashtags): 
                    tweetSocials = socials[currentTweetIndex]
                    print(tweetSocials)
                    if (tweetSocials[0] == "Public_metrics"):
                        break
                    else:
                        parsedSocialInfoClasses[currentNode].AddLikes(tweetSocials[0],tweetSocials[1],tweetSocials[2],tweetSocials[3])
                currentNode += 1
        currentTweetIndex += 1

    #print(currentTweetIndex)
                    
    return parsedSocialInfoClasses

def pearsonCorrelation(centrality):

    likes = []
    values = []


    for tweetclasses in parsedSocialInfoClasses:
        likes.append(tweetclasses.totalLikes)

    values = list(centrality.values())

    print(len(values))
    print(len(likes))


    corr = scipy.stats.pearsonr(likes, values)
    #print(corr[0]) #[0] is pearson correlation


#"{'retweet_count': 34, 'reply_count': 0, 'like_count': 0, 'quote_count': 0}"




if __name__ == "__main__":
    #Transform csv data to dictionary
    #with open('tweetsdata_00.csv',newline='', encoding="utf8") as csvfile:
    #    reader = csv.reader(csvfile)
     #   next(reader)
        # pull in each row as a key-value pair
    #    dictionary_of_tweets = dict(reader)
    
    df = pd.read_csv('merged-csv-files.csv', encoding="utf8", usecols=['Language','Text','Hashtags', 'Public_metrics'])
    
    #df = df['Language'].notnull()
    #df['Text'] = df['Text'].notnull()
    #df = df['hashtags'].notnull()
    #df = df['Public_metrics'].notnull()
    

    #hashtags = df['Hashtags'].to_numpy()
    
    
    
    
    #Draw a histogram showing the popularity of the main hashtags highlighting the number of posts per individual hashtag.
    counted_dict, final = countHashtags(df)
    parsedHashtags, parsedSocialInfoClasses = parseTweets(df)
    parsedSocialMetrics = ParsePublicMetrics(df)

    parsedSocialInfoClasses = FetchSocialAttributes(parsedSocialMetrics, parsedHashtags, parsedSocialInfoClasses)

    #for tweetClasses in parsedSocialInfoClasses:
        #tweetClasses.PrintInfo()


    G = build_hashtag_graph(final, parsedHashtags)
    drawHistogram(counted_dict)
    #Use a pie chart illustrations to show the language of the posts for each of the above main hashtags.
    drawPiechart(df)

    analyzedArray = textAnalyze(df['Text'].to_numpy())
    ternaryplot(analyzedArray)
    
    calculateProperties(G)

    plotLocal(G)
    labelPropagation(G)

    degreeCent = nx.degree_centrality(G)
    eigenCent = nx.eigenvector_centrality(G)
    pageRankCent = nx.pagerank(G)

    pearsonCorrelation(degreeCent)
    pearsonCorrelation(eigenCent)
    pearsonCorrelation(pageRankCent)

