from math import gamma

def age(alpha, d):
    """
    calcul l'espérance de vie d'un véhicule
    pour lequel la probabilité de mort à l'année i est 1 / gamma(1 + alpha * (i + 1)) pour i entre
    1 et d la durée maximale de vie.
    :param alpha:
    :param d:
    :return:
    """
    s = sum([1 / gamma(1 + alpha * (i + 1)) for i in range(d)])
    return sum([(i + 1 / 2) * 1 / gamma(1 + alpha * (i + 1)) for i in range(d)]) / s


def find_alpha(age_moy, duree_vie):
    """
    trouve par dychotomie alpha tq |E[X_alpha] - age_moy|< eps
    où P(X_alpha =i) ~ 1/gamma(1+alpha(1+i))
    alpha == 1 :  P(X_alpha =i) = (1+i)!
    :param age_moy:
    :param duree_vie:
    :return:
    """
    d = 2 * duree_vie
    alpha0 = 0
    alpha1 = 1
    eps = 1e-3
    while age(alpha1, d) > age_moy:
        alpha1 += 1
    while age(alpha0, d) > age_moy + eps:
        alpha = (alpha0 + alpha1) / 2
        if age(alpha, d) > age_moy:
            alpha0 = alpha
        else:
            alpha1 = alpha
    return alpha0

find_alpha(1,40)