🎬 Movie Review Sentiment Analyzer

A complete sentiment analysis project that uses VADER and Logistic Regression to analyze movie reviews, generate insights, and display them on a webpage.

🚀 Features

✅ Cleaned and analyzed movie review dataset
✅ Computed VADER sentiment scores (Positive / Negative / Neutral)
✅ Trained Logistic Regression classifier
✅ Generated dataset-level statistics
✅ Visualized insights and results on a custom HTML webpage

📊 Dataset Details

The project uses a movie review dataset (movie_reviews.xlsx) containing:

1. Serial No
2. URLs
3. Year_Released
4. Title
5. IMDB Ids
6. Cast
7. Plot_Summary
8. Review
9. Genre
10. Released Date
11. Age of Content
12. Plot Keywords
13. Duration

🧠 Sentiment Model

Model: Logistic Regression
Features: VADER polarity scores
Target: Binary classification (Positive = 1, Negative/Neutral = 0)
Accuracy: ~85%

🧮 Python Code

Main analysis script: movie_review
Key functionalities:

1. Calculate dataset statistics
2. Compute VADER sentiment percentages
3. Train Logistic Regression model
4. Evaluate model metrics
5. Generate summary for webpage display

💻 Webpage

Folder: web/
movie_new.html: Displays summary statistics and sentiment distribution

You can open web/index.html directly in your browser to view the results.

⚙️ Requirements

All dependencies are listed in requirements.txt.
Install them using: pip install -r requirements.txt

🧭 How to Run

1. Clone the repository: 
git clone https://github.com/isha-gupta604/movie-sentiment-analyzer.git
cd movie-sentiment-analyzer
2. Install dependencies:
3. Run the analysis: python src/movie_review.py
4. Open the webpage: web/movie_new.html

🏗️ Future Enhancements

1. Use deep learning models.
2. Add movie posters dynamically from TMDB API.

📜 License

This project is open source and available under the MIT License.

🙌 Author

Isha Gupta
💼 GitHub: github.com/isha-gupta604
