import tweepy
import configparser
from os.path import exists
import csv
import time

def write_to_csv(data_dict, field_names):
    #check if csv file exists, if not; find one.
    index_counter = 0
    while(True):
        if exists(f'tweetfile_{index_counter:02d}.csv'):
            index_counter += 1
        else:
            break
    try:
        with open(f'tweetfile_{index_counter:02d}.csv', 'w', newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(data_dict)
        print("Data saved to tweets_data_", index_counter)
    except PermissionError:
        print('File opened elsewhere')
    except IOError:
        print("File write error.")

def search_hashtags(api, num_tweets, hashtag):
    field_names = ['ID', 'Username', 'Time','Geo', 'Language', 'Text', 'Hashtags', 'Public_metrics']
    data_dict = []

    i = 0

    print("Searching tweets...")
    for tweet in tweepy.Paginator(api.search_recent_tweets, query=hashtag, 
                              tweet_fields=['context_annotations', 'created_at', 'entities', 'author_id', 'text', 'lang', 'geo', 'public_metrics'], max_results=10).flatten(limit=num_tweets):
        
        #check all hashtags, if none, skip the tweet
        all_hashtags = find_hashtags(str(tweet.text))
        if all_hashtags == 0:
            continue

        #if the tweet has hashtags, add it to the data
        data_dict.append({'ID': i,
                        'Username': tweet.author_id,
                        'Time': tweet.created_at,
                        'Geo': tweet.geo,
                        'Language': tweet.lang,
                        'Text': str(tweet.text),
                        'Hashtags': find_hashtags(str(tweet.text)),
                        'Public_metrics': tweet.public_metrics
                        })
        i += 1 #for numering of columns in csv file.
    
    #write data to csv file
    print("Writing to csv.............")
    write_to_csv(data_dict, field_names)

    
#find all hashtag words from the text, if no hashtags, return 0
def find_hashtags(text):
    hashtags = set(part[1:] for part in text.split() if part.startswith('#')) #if has #
    if len(hashtags) < 1:
        hashtags = 0
    return hashtags


if __name__ == '__main__':
    #initialize client for twitter api
    config = configparser.ConfigParser()
    config.read('config.ini')

    #read bearer token from config

    client = tweepy.Client(config['twitter']['bearer_token'])

    #search tweets with a certain hashtag

    
    #actual searching functionality
    for i in range(4):
        search_hashtags(client, 1500, "#ukrainewar OR #war OR #army OR #military OR #kiev OR #ua OR #specialforces OR #donbass OR #donbasswar OR #airsoft OR #nomockal OR #warukraine OR #tactics OR #azovsea OR #militarystile OR #azov OR #russia OR #donetsk OR #soldiers OR #ukrainenews OR #odessa OR #ukrainianarmy OR #lviv OR #victory OR #nato OR #kyiv OR #militaryukraine OR #news OR #freesentso")
        time.sleep(900)
    
    print("Gathering done.")
    