import graphlab

products = graphlab.SFrame('amazon_baby.gl/')
print products.head(10)

# build the word count vector for each review
products['word_count'] = graphlab.text_analytics.count_words(products['review'])
print products.head(10)

# Categorical view in the browser
graphlab.canvas.set_target('browser')
products['name'].show()
products['rating'].show('Categorical')


## define what is positive (rating >= 4) and negative (rating <= 2)
# ignore all products whose rating is 3
products = products[products['rating'] != 3]
products['rating'].show('Categorical')

# create sentiment column
products['sentiment'] = products['rating'] >= 4
products['sentiment'].show()


#train sentiment 
train_data,test_data = product.random_split(.8,seed=0)
print "train_data: %s, test_data: %s" % (len(train_data), len(test_data))

sentiment_model = graphlab.logistic_classifier.create(train_data,
													  target='sentiment',
													  features=['word_count'],
													  validation_set=test_data)
sentiment_model.evaluate(test_data,metric='roc_curve')
sentiment_model.show(view='Evaluation')

# fine tune the modeling with selected words.
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

# create columns for each row the counts of those selected words.
#lambda 
for word in selected_words:
    products[word] = products['word_count'].apply(lambda word_count_dic: word_count_dic[word] if word in word_count_dic.keys() else 0)
    print 'the sume of this word %s is : %s' % (word, products[word].sum())

print products.head(30)
products.show()

selected_words_model = graphlab.logistic_classifier.create(train_data, 
														target='sentiment',
														features=selected_words, 
														validation_set=test_data)

coeff =  selected_words_model['coefficients'].sort('value', ascending=False)
coeff.print_rows(num_rows=12)

selected_words_model.evaluate(test_data, metric='roc_curve')
selected_words_model.show(view='Evaluation')


# get the data for another product: Baby Trend Diaper Champ
baby_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']

baby_champ_reviews['predicted_sentiment'] = selected_words_model.predict(baby_champ_reviews, output_type='probability')
baby_champ_reviews = baby_champ_reviews.sort('rating', ascending=False)
print selected_words_model.predict(baby_champ_reviews[0:1], output_type='probability')

baby_champ_reviews['predicted_sentiment'] = sentiment_model.predict(baby_champ_reviews, output_type='probability')
baby_champ_reviews = baby_champ_reviews.sort('rating', ascending=False)
print sentiment_model.predict(baby_champ_reviews[0:1], output_type='probability')


