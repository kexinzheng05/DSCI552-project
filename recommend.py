import numpy as np
from PIL import Image
from skimage import color
from pyemd import emd_samples
import sys
import pandas as pd
import ast

# find 12 most popular articles
def MostPopular(purchase):
    articles = [ast.literal_eval(row[1]) for row in purchase]
    articles = [item for elem in articles for item in elem]
    popular = {}

    for article in articles:
        popular[article] = popular.get(article, 0) + 1
    popularIds = sorted(popular, key=popular.get, reverse=True)[:12]

    return popularIds


def main():
    pd.DataFrame(columns=["customer", "predicition"]).to_csv("recommendArticles.csv", index=False)
    # read all customers and their purchase info
    np.set_printoptions(threshold=sys.maxsize)
    dtype_dic = {'article_id': str}
    purchase = pd.read_csv('purchase.csv', dtype=dtype_dic, header=0, usecols=[1, 2]).values.tolist()
    #print("len(purchase)", len(purchase))
    popularIds = MostPopular(purchase)

    dtype_dic = {'article_id': str}
    articles = pd.read_csv('articles.csv', dtype=dtype_dic,usecols=[0, 4]) # assume articles.csv is in the main direcorty
    articles = articles.set_index("article_id")["product_type_name"].to_dict()
    for oneCustomer in range(len(purchase)):
        # get one customer id and purchased articles
        customerId = purchase[oneCustomer][0]
        firstpurchase = ast.literal_eval(purchase[oneCustomer][1])

        print(firstpurchase)
        totalSimilar = []
        alreadyBuy = []
        # for each article, find their similar items
        for i in range(0, len(firstpurchase)):
            similar = {}

            # get image of purchased article
            articleId = firstpurchase[i]
            print(articleId)
            if articleId in alreadyBuy:
                continue
            articleType = articles[articleId].replace("/", " ")
            print(articleType)
            try:
                targetImage = Image.open("images/%s/%s.jpg" % (articleId[:3], articleId)) # assume folder "images" is in the main direcorty
            except IOError:
                print('%s does not have an image'% articleId)
                continue
            targetImage = targetImage.convert('RGB')
            targetImage = targetImage.resize((32, 32))
            targetImage = color.rgb2lab(targetImage).ravel()

            # find all articles in the same cluster
            similarArticles = pd.read_csv(f'D:/Download/csvfile/{articleType}.csv', dtype=dtype_dic, usecols=[1, 2])
            cluster = similarArticles[similarArticles['article_id'] == articleId].values[0][1]
            print("cluster:", cluster)
            sameGroup = similarArticles[similarArticles['cluster'] == cluster]
            print(len(sameGroup))

            # read articles in the same cluster, calculate EMD
            for Id in sameGroup['article_id']:
                path = "images/%s/%s.jpg" % (Id[:3], Id)
                #print(Id)
                image = Image.open(path).convert('RGB')
                image = image.resize((32, 32))
                image = color.rgb2lab(image).ravel()
                dist = emd_samples(image, targetImage)
                similar[Id] = dist
                #print(Id, dist)
            #print(list(similar.keys()))

            # sort all distances
            similarId = sorted(similar, key=similar.get)
            #print(similarId)
            totalSimilar.append(similarId)
            alreadyBuy.append(articleId)
        validArticle = len(alreadyBuy)
        FinalResult = []
        # if all purchased articles have no image
        if validArticle == 0:
            df = pd.DataFrame({"customer": customerId, "predicition": ' '.join(popularIds)}, index=[0])
            df.to_csv('recommendArticles.csv', mode="a", index=False, header=False)
            continue
        if validArticle <= 12:
            eachNum = 12//validArticle
        else:
            eachNum = 1

        # select the most similar articles
        for i in range(validArticle):
            j = 1

            while j <= len(totalSimilar[i]) and j < eachNum+1:
                print(totalSimilar[i][j])
                if totalSimilar[i][j+1] not in FinalResult:
                    FinalResult.append(totalSimilar[i][j])
                j+=1
            #if len(totalSimilar[i]) > 1:
            #FinalResult += totalSimilar[i][1:eachNum+1]

        if len(FinalResult) < 12 and len(totalSimilar[-1]) > eachNum+2:
            FinalResult += totalSimilar[-1][eachNum+1:]
        if len(FinalResult) < 12:
            for index in range(12):
                if popularIds[index] not in FinalResult:
                    FinalResult.append(popularIds[index])

        FinalResult = FinalResult[:12]

        # write to csv file
        FinalResult = ' '.join(FinalResult)
        print(FinalResult)
        df = pd.DataFrame({"customer": customerId, "predicition": FinalResult}, index=[0])
        df.to_csv('recommendArticles.csv', mode="a", index=False, header=False)


if __name__ == "__main__":
    main()
