import numpy as np
import re
from collections import defaultdict

class Polynomial:
    number_types = (int, float, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64,
                          np.float16, np.float32, np.float64)
    int_type = (int, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)
    style = 'unicode'

    def __init__(self, p=np.array([0])):
        if isinstance(p, str):
            self.init_str(p)
        else:
            self.init_array(p)

    def init_str(self, s):
        self.ls = PolynomialStringParser.parse(s)

    def init_array(self, ls):
        # coefficient array
        ls = np.array(ls)
        if ls.shape[0] == 0:
            ls = np.array([0])
        elif np.all(ls == 0):
            ls = np.array([0])

        self.ls = np.array(
            ls[:ls.shape[0] - np.argmax(np.flip(ls) != 0)]
        )

    def __getitem__(self, index):
        if index <= self.deg():
            return self.ls[index]
        else: return 0
    
    def __call__(self, x):
        # deals with all sorts of shit, array or number, all OK, even return same "type"
        # most of the operation is performed to deal with arrays

        # x: (...)
        x = np.array(x)

        # xs: (deg+1, ...), basically x copied deg+1 times
        xs = np.ones((self.deg()+1, *x.shape)) * x

        # power: (deg+1, ...), with power[i] = x**i
        x_powers = np.power(xs, np.arange(self.deg()+1).reshape((self.deg()+1, *([1] * len(x.shape)))))

        # ret: (...), the result of P(x)
        ret = np.sum(self.ls.reshape((self.deg()+1, *([1] * len(x.shape)))) * x_powers, axis=0)
        return ret
            
    def copy(self):
        return Polynomial(self.getCoefficient())

    def getCoefficient(self):
        return np.copy(self.ls)
    def deg(self):
        return self.ls.shape[0] - 1
    def padToDeg(self, deg):
        # pad the array to desired degree
        if self.deg() >= deg:
            return np.copy(self.ls)
        else:
            return np.concatenate((self.ls, np.zeros(deg - self.deg())))

    def __add__(self, p):
        max_deg = max(self.deg(), p.deg())
        return Polynomial(self.padToDeg(max_deg) + p.padToDeg(max_deg))
    def __iadd__(self, p):
        self.ls = self.__add__(p).getCoefficient()
        return self

    def __sub__(self, p):
        return self.__add__(-p)
    def __isub__(self, p):
        self.ls = self.__add__(-p).getCoefficient()
        return self
    def __neg__(self):
        return Polynomial(-self.ls)
    def __eq__(self, p):
        max_deg = max(self.deg(), p.deg())
        return np.all(np.isclose(self.padToDeg(max_deg), p.padToDeg(max_deg)))
    def __ne__(self, p):
        return not self.__eq__(p)

    
    def __lshift__(self, value):
        # defined as multiply by x^value, just like numbers
        value = int(value)
        return Polynomial(np.concatenate((np.zeros(value), self.ls)))
    def __ilshift__(self, value):
        self.ls = self.__lshift__(value).getCoefficient()
        return self

    def __rshift__(self, value):
        # defined as divide by x^value, without remainder
        value = int(value)
        if value < 0: raise Exception("rshift with value < 0")
        if self.deg() >= value:
            return Polynomial(self.ls[value:])
        else:
            return Polynomial([0])
    def __irshift__(self, value):
        self.ls = self.__rshift__(value).getCoefficient()
        return self

    def __mul__(self, p):
        if isinstance(p, Polynomial.number_types):
            return Polynomial(self.ls * p)

        # assumes polynomial
        result = Polynomial([0])
        for deg, coef in enumerate(self.ls):
            result += (p << deg) * coef
        return result

    def __imul__(self, p):
        self.ls = self.__mul__(p).getCoefficient()
        return self
    def __rmul__(self, p):
        return self.__mul__(p)
    
    def __pow__(self, val):
        if not isinstance(val, Polynomial.int_type) or val < 0:
            raise Exception(f'Value not valid for power: {val}')
        val = int(val)
        ret = Polynomial([1])
        power = self.copy()
        while val != 0:
            if val & 1:
                ret *= power
            val >>= 1
            power *= power
        return ret

    def PolynomialDivision(self, p):
        # do a proper full division, return (d, r)
        if self.deg() < p.deg():
            return (Polynomial([0]), Polynomial(self.getCoefficient()))
        divider = Polynomial(self.getCoefficient())
        stack = []
        for shift in range(self.deg() - p.deg(), -1, -1):
            cur_result_coef = divider[p.deg() + shift] / p[-1]
            stack.append(cur_result_coef)
            divider -= (p << shift) * cur_result_coef
        stack.reverse()
        return (Polynomial(stack), Polynomial(divider.getCoefficient()[:p.deg()+1]))

    def __truediv__(self, p):
        # same as PolynomialDivision
        if isinstance(p, Polynomial.number_types):
            return Polynomial(self.ls / p)
        return self.PolynomialDivision(p)
    
    def __itruediv__(self, p):
        if not isinstance(p, Polynomial.number_types):
            raise Exception('Cannot itruediv with non-number type')
        self.ls = self.__truediv__(p).getCoefficient()
        return self

    def __floordiv__(self, p):
        if isinstance(p, Polynomial.number_types):
            return Polynomial(self.ls / p)
        return self.PolynomialDivision(p)[0]

    def __ifloordiv__(self, p):
        self.ls = self.__floordiv__(p).getCoefficient()
        return self

    def __mod__(self, p):
        # only supports polynomial, for obvious reason
        return self.PolynomialDivision(p)[1]
    def __imod__(self, p):
        self.ls = self.__mod__(p).getCoefficient()
        return self
    
    def deriv(self):
        if self.deg == 0:
            return Polynomial([0])
        
        return Polynomial(self.ls[1:] * (np.arange(self.deg()) + 1))
    
    @classmethod
    def setPrintStyle(cls, style):
        cls.style = style

    def __str__(self):
        def getSuperscript(num):
            if num < 2: return ''
            return str(num).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))
        str_ls = []
        for i, coef in enumerate(self.ls):
            if coef == 0: continue
            s = ''
            if len(str_ls) != 0: 
                s += (' + ' if coef > 0 else ' - ')
            elif coef < 0:
                s += '-'
            s += f'{abs(coef)}'
            if i > 0:
                if Polynomial.style == 'unicode': s += f'⋅x{getSuperscript(i)}'
                else: s += f'*x^{i}'
            str_ls.append(s)
        if len(str_ls) == 0: return '0'
        else: return ''.join(str_ls)
    def __repr__(self):
        return self.__str__()


# parses polynomial string
class PolynomialStringParser:
    """
        parse polynomial strings, defined as:
        - ignore all spaces
        - can read:
            - '-7', '3', '+3x', '-3*x', '3x^2', '3*x^3', 'x^2', 'x', '-x' and all combinations
            - e.g. '7x^3+6x-4+5*x^700+x+5+7'
        - you can write '3+*x' and it will read as '3+x', oh well
    """
    @classmethod
    def parse(cls, s):
        # returns {power: coefficient}
        s = s.replace(' ', '')

        match_ls = cls.findAllParts(s)
        coef_d = defaultdict(lambda: 0)
        coef_d[0] = 0   # initialize
        for match in match_ls:
            part_result = cls.parseOnePart(match)
            coef_d[part_result[0]] += part_result[1]
        
        return cls.makePolynomialList(coef_d)
    
    @classmethod
    def findAllParts(cls, s):
        part_pat = r"([+-]?[.0-9]*)([*]?x(\^([0-9]+))?)?"
        all_pat = f"^({part_pat})*$"
        if re.match(all_pat, s) == None:
            raise Exception('Polynomial parser got a string that cannot be parsed!')
        return re.findall(part_pat, s)

    @classmethod
    def parseOnePart(cls, part_ls):
        # returns [power, coefficient]
        if cls.isEmpty(part_ls):
            return (0, 0)  # is a must, since ('', '', '', '') will yield (0, 1) below
        return (cls.getPower(part_ls), cls.getCoef(part_ls))

    @classmethod
    def isEmpty(cls, ls):
        return all([elem == '' for elem in ls])
    
    @classmethod
    def getPower(cls, part_ls):
        if part_ls[1] == '': return 0
        elif part_ls[3] == '': return 1
        else: return int(part_ls[3])

    @classmethod
    def getCoef(cls, part_ls):
        if part_ls[0] == '': return 1
        if part_ls[0] == '+': return 1
        elif part_ls[0] == '-': return -1
        else: return float(part_ls[0])

    @classmethod
    def makePolynomialList(cls, coef_d):
        max_deg = max(coef_d.keys())
        ls = np.zeros(max_deg+1)
        for i, coef in coef_d.items():
            ls[i] += coef
        return ls

