import graphlab
people = graphlab.SFrame('people_wiki.gl/')
people.head()
len(people)

#exam obanma data 
obama = people[people['name'] == 'Barack Obama']
obama['word_count'] = graphlab.text_analytics.count_words(obama['text'])
print obama['word_count']
obama_word_count_table = obama[['word_count']].stack('word_count', new_column_name = ['word','count'])
obama_word_count_table.head()
obama_word_count_table.sort('count',ascending=False)

#using TF-IDF on corpus 
people['word_count'] = graphlab.text_analytics.count_words(people['text'])
people.head()
tfidf = graphlab.text_analytics.tf_idf(people['word_count'])

if graphlab.version <= '1.6.1':
    tfidf = tfidf['docs']

people['tfidf'] = tfidf

#implementing TF-IDF
obama[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)
clinton = people[people['name'] == 'Bill Clinton']
beckham = people[people['name'] == 'David Beckham']
graphlab.distances.cosine(obama['tfidf'][0],clinton['tfidf'][0])
graphlab.distances.cosine(obama['tfidf'][0],beckham['tfidf'][0])

#KNN for document retrieval 
knn_model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name')
knn_model.query(obama)

# try other celebrities
swift = people[people['name'] == 'Taylor Swift']
knn_model.query(swift)
jolie = people[people['name'] == 'Angelina Jolie']
knn_model.query(jolie)


elton = people[people['name']=='Elton John']
elton['word_count']=graphlab.text_analytics.count_words(Elton['text'])
print elton['word_count']
elton_count_table = elton[['word_count']].stack('word_count',new_column_name=['word','count'])
elton_count_table.head()
elton_count_table.sort('count',ascending=False)
elton[['tfidf']].stack('tfidf',new_column_name=['word','tfidf']).sort('tfidf',ascending=False)

#raw word count 
knn_model_wordcount = graphlab.nearest_neighbors.create(people,features=['word_count'],label='name',distance='cosine')
knn_model_wordcount.query(elton)
knn_model = graphlab.nearest_neighbors.create(people,features=['tfidf'],label='name',distance='cosine')
knn_model.query(elton)