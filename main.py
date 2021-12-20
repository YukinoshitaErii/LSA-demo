import time

import process_data as pre
import pLSA

if __name__ == '__main__':
    localtime = time.time()
    item_user, c = pre.train()          # train LSA
    plsa = pLSA.PLSA(item_user, c)      # get a pLSA model
    pLSA.train(plsa)                    # train plsa
    d, v = pre.main()                   # do prediction
    print('-----------------------------------------------------------------------------------------------')
    print(f' LSA result is {d}')
    localtime_I = time.time()
    print(f"cost {localtime_I-localtime}")



