import sys
import numpy as np
import pandas as pd

# get customer ids and their purchased products
def main():
    np.set_printoptions(threshold=sys.maxsize)
    purchase = {}
    dtype_dic = {'article_id': str}
    purchases = pd.read_csv('transactions_train.csv', dtype=dtype_dic, header=0, usecols=[1, 2]) # assume transactions_train.csv is in the main direcorty
    purchases = purchases.values.tolist()[:1000]
    for customer, article in purchases:
        purchase.setdefault(customer, []).append(article)
    print(purchase)

    df = pd.DataFrame({"customer_id": list(purchase.keys()), "article_id": list(purchase.values())})
    df.to_csv("purchase.csv")


if __name__ == '__main__':
    main()
