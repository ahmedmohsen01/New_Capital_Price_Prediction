🏙️ Real Estate Price Prediction in Egypt 🇪🇬
This project is a property price prediction tool and analytical dashboard for residential listings in Egypt, with a focus on the New Administrative Capital. Built using Streamlit, scikit-learn, and Pandas, the application provides users with:

🔍 Features
Machine Learning Model (Random Forest) that predicts property prices based on:

Property type

Location

Apartment area (sqm)

Number of bedrooms and bathrooms

Selected amenities

Interactive Streamlit App with:

A clean UI and banner of Egypt’s New Capital

Dropdown inputs for property type and location to avoid invalid inputs

Real-time price prediction with visual feedback

Dashboard displaying:

Average apartment price

Average price per square meter

Minimum and maximum property prices

Price distribution histogram

Outlier handling:

Removes unrealistic entries (e.g., prices above 500M EGP or below 3M EGP) for better model accuracy

📦 Files Included
app.py: Streamlit application

model.pkl: Trained Random Forest regression model

features.pkl: Feature columns after one-hot encoding

data.csv: Cleaned and filtered dataset (used for analytics)

📈 Use Case
This tool helps buyers, real estate analysts, and developers estimate property values and gain insights into market trends based on data scraped from online property listings.

Let me know if you want to include example screenshots, deployment instructions (e.g., Streamlit Cloud), or badges (like “made with ❤️ in Egypt”).
