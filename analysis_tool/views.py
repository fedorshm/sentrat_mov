from django.shortcuts import render
from transformers import pipeline

sentiment_model = pipeline("text-classification", model="Gnider/roberta_dist_sent", truncation=True)
rating_model = pipeline("text-classification", model="Gnider/roberta_dist_rat", truncation=True)

def chunk_text(text, max_length=512):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def aggregate_results(sentiment_chunks, rating_chunks):
    sentiment_scores = {'positive': 0, 'negative': 0}
    for chunk in sentiment_chunks:
        for score in chunk:
            if score['label'] == 'LABEL_1':  
                sentiment_scores['positive'] += score['score']
            elif score['label'] == 'LABEL_0':
                sentiment_scores['negative'] += score['score']

    final_sentiment = 'positive' if sentiment_scores['positive'] > sentiment_scores['negative'] else 'negative'

    rating_aggregate = {i: 0 for i in range(8)} 
    label_to_rating = {0: 1, 1: 2, 2: 3, 3: 4, 4: 7, 5: 8, 6: 9, 7: 10}

    for chunk in rating_chunks:
        for score in chunk:
            label = int(score['label'].split('_')[1])
            rating_aggregate[label] += score['score']

    sorted_ratings = sorted(rating_aggregate.items(), key=lambda item: item[1], reverse=True)[:3]
    top_ratings = [{"rating": label_to_rating[label], "score": score} for label, score in sorted_ratings]

    final_rating = label_to_rating[sorted_ratings[0][0]]

    return sentiment_scores, final_sentiment, top_ratings, final_rating

def analyze_text(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')

        text_chunks = chunk_text(text)

        sentiment_chunks = [sentiment_model(chunk, return_all_scores=True)[0] for chunk in text_chunks]
        rating_chunks = [rating_model(chunk, return_all_scores=True)[0] for chunk in text_chunks]

        sentiment_scores, final_sentiment, top_ratings, final_rating = aggregate_results(sentiment_chunks, rating_chunks)

        return render(request, 'analysis_tool/results.html', {
            'text': text,
            'sentiment_scores': sentiment_scores,
            'top_ratings': top_ratings,
            'final_sentiment': final_sentiment,
            'final_rating': final_rating
        })

    return render(request, 'analysis_tool/analyze_text.html')

