from datasets import load_dataset
from json import dumps 

topicnames = [
    'Computing', 'Education', 'Mathematics', 'Music', 'Meteorology', 'Media', 'Biology', 'History', 'Transport', 'Law', 'Literature', 'Art', 'Chemistry', 'Economics', 'Engineering', 'Finance', 'Geology', 'Government', 'Heraldry', 'Language', 'Medicine', 'Opera', 'Physics', 'Politics', 'Recreation', 'Religion', 'Technology', 'War', 'Society', 'Business', 'Astronomy', 'Geophysics', 'Spaceflight', 'Health', 'Military', 'Philosophy', 'Archaeology', 'Geography', 'Culture', 'Theatre', 'Architecture', 'Linguistics', 'Military history', 'Nobility', 'Communication', 'Drink', 'Death', 'Ethics', 'Entertainment', 'Energy', 'Food', 'Internet', 'Life', 'Metaphysics', 'Mass media', 'Nature', 'Science', 'Spirituality', 'Time', 'Universe', 'Humanities', 'Knowledge', 'Human behavior', 'People', 'Information', 'Love', 'Sex'
]

topics = dict()
data = load_dataset("wikipedia", "20220301.en", streaming=True)
for page in data['train']:
    if page['title'] in topicnames:
        print(page['title'])
        topics[page['title']] = page['text']
    if len(topics)==len(topicnames):
        break
with open('embeddings/wikipedia_documents.json','w') as f:
    f.write(dumps(topics,indent=2))
