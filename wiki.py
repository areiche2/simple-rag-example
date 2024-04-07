import requests


def fetch(title):
    return requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            'action': 'query',
            'prop': 'revisions',
            'rvlimit': 1,
            'titles': title,
            'rvslots': '*',
            'rvprop': 'content',
            'formatversion': 2,
            'format': 'json',

        }
    ).json()["query"]["pages"][0]["revisions"][0]['slots']['main']['content']
