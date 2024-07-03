from decimal import Decimal, getcontext
from matplotlib.pyplot import imshow, show, plot
from numpy import zeros
from struct import pack


def pi(decimal_places:int) -> str:
    getcontext().prec = decimal_places
    res = Decimal(0)
    for i in range(decimal_places):
        a = Decimal(1)/(16**i)
        b = Decimal(4)/(8*i+1)
        c = Decimal(2)/(8*i+4)
        d = Decimal(1)/(8*i+5)
        e = Decimal(1)/(8*i+6)
        r = a*(b-c-d-e)
        res += r
    print(float(res))
    return str(res)


def float_to_binary(number:float) -> str:
    return ''.join('{:0>8b}'.format(c) for c in pack('!f', number))


def logistic_map(digits_of_pi:str) -> None:
    sequence = list(map(int,digits_of_pi))
    plot(sequence[:-1],sequence[1:])
    show()

def space_time_evolution(digits_of_pi:str) -> None:
    T = len(digits_of_pi)
    evolution = zeros(shape=(T,10))
    for row in range(T):
        digit = int(digits_of_pi[row])
        evolution[row,:][digit] = 1
    imshow(evolution, cmap='gray')
    show()


T = 100
digits_of_pi = pi(decimal_places=T+2)[2:]

#https://pastebin.com/z2er2ET7
#https://mathworld.wolfram.com/PiDigits.html