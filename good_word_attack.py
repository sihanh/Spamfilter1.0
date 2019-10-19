import random
from spam_sky import nb_predict
from spam_maxent_api import ma_predict

best_n_word = ['lor','say','happi','tone','morn','co','sleep','wat','finish','went','watch','lol', 'anyth','someth','gon','always','amp','dun','bit','sure']
first_n_word = ['zyada','zouk','zoom','zogtorius','zoe','zindgi','zhong','zed','zeros','zaher','yupz','yup','yuo','yun','ystrday','vrs','yr','yowifes','youuuu','yoga']

def word_attack_NB(word_list, message):
    # Initial prediction of the input message
    nb = nb_predict(message)
    i_nb = 0

    while nb is 'spam':
        r = word_list[random.randint(0,19)]
        message = message + ' ' + r
        nb = nb_predict(message)
        i_nb = i_nb + 1
    return i_nb

def word_attack_MA(word_list, message):
    # Initial prediction of the input message
    ma = ma_predict(message)
    i_ma = 0

    while ma is 'spam':
        r = word_list[random.randint(0, 19)]
        message = message + ' ' + r
        ma = ma_predict(message)
        i_ma = i_ma + 1
    return i_ma

