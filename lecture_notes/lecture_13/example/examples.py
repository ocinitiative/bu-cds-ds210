import varia.multiply as mult
from varia.add import double

import varia.emotions as emo

if __name__ == "__main__":
    x = 7
    y = 9
    print(x,"times",y,"is",mult.multiply(x,y))
    print('doubled',x,'is',double(x))
    
    print('====================')
    
    s = 'angry Bob'
    t = 'surprised Alice'
    
    print(s,'and',t)
    
    s = emo.make_sad(s)
    t = emo.make_happy(t)
    
    print(s,'and',t)
