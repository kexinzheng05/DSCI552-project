DSCI552-project
 
In this project, the datasets from a Kaggle competition “H&M Personalized Fashion Recommendations” is used.
The website is https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview
It contains a folder of images, articles.csv, and transactions_train.csv

This project is to build a fashion recommendation program. 

It combined FastMap and K-means to divided products into different groups according to their HOG images, then used Earth Mover's Distance (EMD) to obtain the products with similar colors, and finally picked 12 products for each customer.
