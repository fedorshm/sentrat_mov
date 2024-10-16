from django.shortcuts import render
from django.http import JsonResponse
from transformers import pipeline

sentiment_model = pipeline("text-classification", model="Gnider/roberta_dist_sent", truncation=True, max_length=512)
rating_model = pipeline("text-classification", model="Gnider/roberta_dist_rat", truncation=True, max_length=512)

def analyze_text(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')

        if len(text) > 512:
            text = text[:512]

        sentiment_results = sentiment_model(text, return_all_scores=True)[0]
        label_to_sentiment = {'LABEL_0': 'negative', 'LABEL_1': 'positive'}
        sentiment_scores = {label_to_sentiment[res['label']]: res['score'] for res in sentiment_results}

        rating_results = rating_model(text, return_all_scores=True)[0]
        sorted_ratings = sorted(rating_results, key=lambda x: x['score'], reverse=True)[:3]
        label_to_rating = {0: 1, 1: 2, 2: 3, 3: 4, 4: 7, 5: 8, 6: 9, 7: 10}
        top_ratings = [{"rating": label_to_rating[int(res['label'].split('_')[1])], "score": res['score']} for res in sorted_ratings]

        if sentiment_scores['positive'] > sentiment_scores['negative']:
            final_sentiment = "positive"
        else:
            final_sentiment = "negative"

        final_rating = top_ratings[0]['rating']

        return render(request, 'analysis_tool/results.html', {
            'text': text,
            'sentiment_scores': sentiment_scores,
            'top_ratings': top_ratings,
            'final_sentiment': final_sentiment,
            'final_rating': final_rating
        })
    
    return render(request, 'analysis_tool/analyze_text.html')
