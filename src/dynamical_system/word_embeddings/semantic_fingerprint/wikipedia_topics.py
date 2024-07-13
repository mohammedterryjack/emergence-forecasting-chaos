from datasets import load_dataset
from json import dumps 

topicnames = []

topics = []
data = load_dataset("wikipedia", "20220301.en", streaming=True)
for page in data['train']:
    if page['title'] in topicnames:
        print(page['title'])
        topics.append(
            {
                'title':page['title'],
                'text':page['text']
            }
        )
with open('embeddings/contexts.json','w') as f:
    f.write(dumps(topics,indent=2))