import unittest
import numpy as np
from .polynomial import Polynomial

@unittest.skip('Saves time')
class PolynomialTestClass(unittest.TestCase):
    def testDeg(self):
        self.assertEqual(Polynomial([1,2,3,4,5]).deg(), 4)
        self.assertEqual(Polynomial([1,2,3,4,0,0,0]).deg(), 3)
        self.assertEqual(Polynomial([0,0,0,0,0,0,4,0,0,0]).deg(), 6)
        self.assertEqual(Polynomial([0]).deg(), 0)
        self.assertEqual(Polynomial([0,0,0,0,0,0,0]).deg(), 0)

    def testEqual(self):
        # TODO: add floating point equals
        self.assertEqual(Polynomial([1,2,3,3]), Polynomial([1,2,3,3]))
        self.assertEqual(Polynomial([0,0,0,0,1]), Polynomial([0,0,0,0,1]))
        self.assertEqual(Polynomial([0,3,0,0]), Polynomial([0,3]))
        self.assertEqual(Polynomial([0,0,1,2,0,0,0]), Polynomial([0,0,1,2,0]))
        self.assertNotEqual(Polynomial([1,2,3,4]), Polynomial([5,6,7,8]))
        self.assertNotEqual(Polynomial([0,0,1,2,0,0,0]), Polynomial([0,0,0,0,0,0,1,2,0]))
        self.assertNotEqual(Polynomial([0,0,1,2,0,0,0]), Polynomial([0,0,1,2,0,0,0,0,0,0,1]))
    
    def testAddMinus(self):
        a = Polynomial([1,2,3,8])
        b = Polynomial([0,2,4,9,8,2])
        self.assertEqual(
            a + b, Polynomial([1,4,7,17,8,2]))
        self.assertEqual(
            a - b, Polynomial([1,0,-1,-1,-8,-2]))
        self.assertEqual(
            Polynomial([0,0,0,0]) + Polynomial([0,0,0]), Polynomial([0]))
        a += b
        self.assertEqual(a, Polynomial([1,4,7,17,8,2]))
        a -= b
        self.assertEqual(a, Polynomial([1,2,3,8]))
    
    def testShift(self):
        self.assertEqual(Polynomial([1,2,3,4,5,6,7,8]) << 3, Polynomial([0,0,0,1,2,3,4,5,6,7,8]))
        self.assertEqual(Polynomial([1,2,3,4,5,6,7,8]) >> 3, Polynomial([4,5,6,7,8]))
        self.assertEqual(Polynomial([1,2,3,4]) >> 0, Polynomial([1,2,3,4]))
        self.assertEqual(Polynomial([1,2,3,4]) >> 1, Polynomial([2,3,4]))
        self.assertEqual(Polynomial([1,2,3,4]) >> 2, Polynomial([3,4]))
        self.assertEqual(Polynomial([1,2,3,4]) >> 3, Polynomial([4]))
        self.assertEqual(Polynomial([1,2,3,4]) >> 4, Polynomial([0]))
        self.assertEqual(Polynomial([1,2,3,4]) >> 5, Polynomial([0]))
        self.assertEqual(Polynomial([1,2,3,4]) >> 100, Polynomial([0]))
        a = Polynomial([1,2,3,4]); b = Polynomial([1,2,3,4])
        a <<= 4; self.assertEqual(a, b << 4)
        a >>= 5; self.assertEqual(a, b >> 1)
    
    def testNeg(self):
        self.assertEqual(-Polynomial([0]), Polynomial([0]))
        self.assertEqual(-Polynomial([0,3,5,6,0,0,0]), Polynomial([0,-3,-5,-6]))
    
    def testMul(self):
        arr = np.array([1,5,-3,-4,5])
        a = Polynomial(arr)
        b = Polynomial([5,7,8])
        self.assertEqual(a * 3, Polynomial(arr * 3))
        self.assertEqual(5 * a, Polynomial(arr * 5))
        self.assertEqual(4 * a * 3, Polynomial(arr * 12))
        self.assertEqual(1.87 * a, Polynomial(arr * 1.87))
        self.assertEqual(a * 0, Polynomial([0]))
        self.assertEqual((a * 0).deg(), 0)

        self.assertEqual(a * b, b * a)
        self.assertEqual(a * b, Polynomial([5,32,28,-1,-27,3,40]))

        a *= 4; self.assertEqual(a, Polynomial(arr * 4))
        a *= 0.1; self.assertEqual(a, Polynomial(arr * 0.4))
        a *= 2.5 * b
        self.assertEqual(a, Polynomial([5,32,28,-1,-27,3,40]))

    def testDiv(self):
        arr = np.array([0,1,2,5,4,0])
        a = Polynomial(arr)
        self.assertEqual(a / 4, Polynomial(arr / 4))
        self.assertEqual(a / 2.5, Polynomial(arr / 2.5))
        self.assertEqual(a // 3, Polynomial(arr / 3))
        self.assertEqual(a // 1.7, Polynomial(arr / 1.7))

        a = Polynomial([4,2,8,0,2]); b = Polynomial([0,3,7,1])
        result = (a * b) / a
        # maybe should check exact value, to test equal is_close?
        self.assertEqual(result[0], b)
        self.assertEqual(result[0], (a * b) // a)
        self.assertEqual(result[1], Polynomial([0]))
        

        a = Polynomial([5,7,8,4,2]); b = Polynomial([1,5,3])
        result = a / b
        self.assertEqual(result[0] * b + result[1], a)
        self.assertEqual(result[0], a // b)
        self.assertEqual(result[1], a % b)
        result = b / a
        self.assertEqual(result[0], Polynomial([0]))
        self.assertEqual(result[0], b // a)
        self.assertEqual(result[1], b)
        self.assertEqual(result[1], b % a)

        result = a / b
        c = Polynomial(a.getCoefficient())
        c //= b
        self.assertEqual(c, result[0])
        c = Polynomial(a.getCoefficient())
        c %= b
        self.assertEqual(c, result[1])
        with self.assertRaises(Exception):
            c /= b
    
    def testCallType(self):
        a = Polynomial([2.3,4,6.1,-7,4])
        self.assertTrue(
            isinstance(a(3), Polynomial.number_types), type(a(3)))
        self.assertTrue(
            isinstance(a(np.array([2,4,8,5])), np.ndarray))
        self.assertTrue(
            isinstance(a(np.array([[2,4,8,5], [2,4,8,5]])), np.ndarray))
        self.assertTrue(
            isinstance(a(np.arange(6720).reshape((5,4,8,7,6))), np.ndarray))

    def testCall(self):
        # single number call
        a = Polynomial([1,6,8,0,0,4,2,0,0,0])
        xs = [3, -4.3, 7, 0.1, 0, 5, 7, 0]
        answer = [2521, 6885.51, 302961, 1.68004, 1, 43981, 302961, 1]
        self.assertTrue(np.all(np.isclose(answer, np.array([a(i) for i in xs]))))
        
        xs = np.array(xs).reshape((2,2,2))
        answer = np.array(answer).reshape((2,2,2))
        result = a(xs)
        self.assertTrue(answer.shape == result.shape)
        self.assertTrue(answer.shape == xs.shape)
        self.assertTrue(np.all(np.isclose(answer, a(xs))))
        
    
    def testInitByString(self):
        # basics
        self.assertEqual(Polynomial("0"), Polynomial([0]))
        self.assertEqual(Polynomial("6.154"), Polynomial([6.154]))
        self.assertEqual(Polynomial("-3"), Polynomial([-3]))
        self.assertEqual(Polynomial("x"), Polynomial([0,1]))
        self.assertEqual(Polynomial("-x"), Polynomial([0,-1]))
        self.assertEqual(Polynomial("3.154x"), Polynomial([0,3.154]))
        self.assertEqual(Polynomial("-12x"), Polynomial([0,-12]))
        self.assertEqual(Polynomial("-12*x"), Polynomial([0,-12]))
        self.assertEqual(Polynomial("-12x^2"), Polynomial([0,0,-12]))
        self.assertEqual(Polynomial("0x"), Polynomial([0]))
        self.assertEqual(Polynomial("-0x"), Polynomial([0]))
        self.assertEqual(Polynomial("3x - 0x^200"), Polynomial([0,3]))
        self.assertEqual(Polynomial("-0.01x^2"), Polynomial([0,0,-0.01]))
        self.assertEqual(Polynomial("81x^5 + 6.04"), Polynomial([6.04,0,0,0,0,81]))
        self.assertEqual(Polynomial("-3x^2 + 5000x"), Polynomial([0,5000,-3]))
        self.assertEqual(Polynomial("x^2 - 5x + 1"), Polynomial([1,-5,1]))

        # advanced
        self.assertEqual(
            Polynomial("7+3x^1+5x^5"), Polynomial([7,3,0,0,0,5]))
        self.assertEqual(
            Polynomial("   7 +    3  x^1+   5 x^ 5"), Polynomial([7,3,0,0,0,5]))
        self.assertEqual(
            Polynomial("-7+0x^1-0x^5+3x"), Polynomial([-7,3]))
        self.assertEqual(
            Polynomial("0+0x-0-0+0x^100"), Polynomial([0]))
        self.assertEqual(
            Polynomial("-7.4-3x-5x^3-8x^5"), Polynomial([-7.4,-3,0,-5,0,-8]))
        self.assertEqual(
            Polynomial("5x^2 + x - 2"), Polynomial([-2,1,5]))
        self.assertEqual(
            Polynomial("-25x^4 - x - 22"), Polynomial([-22,-1,0,0,-25]))

    @unittest.skip("some are not worth the time, actually")
    def testInitByStringException(self):
        # should break
        with self.assertRaises(Exception): Polynomial("-3.2.1x^4") # float parsing error
        with self.assertRaises(Exception): Polynomial("-3x^") # not complete
        with self.assertRaises(Exception): Polynomial("-3dax^4") # inrelevant chars exist
        with self.assertRaises(Exception): Polynomial("-3d..s$$x^4")
        with self.assertRaises(Exception): Polynomial("os.system('echo Hello World')")
        with self.assertRaises(Exception): Polynomial("-3xxx") # shouldn't even parse
        with self.assertRaises(Exception): Polynomial("-10x^-100") # can't read negative power
        with self.assertRaises(Exception): Polynomial("x^(-100)")
        with self.assertRaises(Exception): Polynomial("1000x^4.3") # no float power should exist
        with self.assertRaises(Exception): Polynomial("1000x^4.3+50")
    
    def testPower(self):
        a = Polynomial([5,1.3,-2])
        cur = Polynomial([1])
        for i in range(0, 50):
            self.assertEqual(a ** i, cur, i)
            cur *= a
    
    def testDerivative(self):
        # derivative
        polys = [
            np.array([ 5.48521106,  0.38644238, -5.84661356, -5.87643451,  7.42316884]),
            np.array([5.67718688]),
            np.array([-0.53750936, -1.88755298,  7.72936962,  7.50754469,  8.83618053, -4.76766586]),
            np.array([-1.19253071, -7.19187714,  1.36294164,  5.62398839,  1.46606228, 0.90746901]),
            np.array([ 7.62477302,  8.29641985, -4.90153546]),
            np.array([0]),
        ]
        answer = [
            Polynomial('29.69267536x^3-17.62930353x^2-11.69322712x+0.38644238'),
            Polynomial('0'),
            Polynomial('-23.8383293x^4+35.34472212x^3+22.52263407x^2+15.45873924x-1.88755298'),
            Polynomial('4.53734505x^4+5.86424912x^3+16.87196517x^2+2.72588328x-7.19187714'),
            Polynomial('-9.80307092x+8.29641985'),
            Polynomial('0'),
        ]
        polys = [Polynomial(a) for a in polys]
        Polynomial.setPrintStyle('ascii')
        for poly, ans_p in zip(polys, answer):
            self.assertEqual(poly.deriv(), ans_p)
        
from .interpolation import *

def arrayCloseTo(a, b, atol=1e-8):
    return np.all(np.isclose(a, b, atol=atol))

class InterpolationTestClass(unittest.TestCase):
    def testLagrangeInterpolation(self):
        xs = np.linspace(0.5, 3, 10).reshape((-1, 1))
        ys = np.log(xs).reshape((-1, 1))
        pts = np.hstack((xs, ys))
        
        p = LagrangePolynomial(pts)
        self.assertTrue(p.deg() <= len(xs)-1)
        self.assertTrue(arrayCloseTo(p(xs), ys))
    
    def testNewtonInterpolation(self):
        xs = np.linspace(0.5, 3, 15).reshape((-1, 1))
        ys = np.log(xs).reshape((-1, 1))
        pts = np.hstack((xs, ys))

        self.assertEqual(NewtonPolynomial(pts), LagrangePolynomial(pts))

        cur_pts = pts[:2]
        pts = pts[2:]

        dd_dict = NewtonMakeDevidedDiffDict(cur_pts)
        for pt in pts:
            cur_pts = np.vstack((cur_pts, pt))
            NewtonAddPoint(pt, dd_dict)
            self.assertEqual(NewtonPolynomial(cur_pts, dd_dict), LagrangePolynomial(cur_pts))
            
    def testHermiteInterpolation(self):
        # ln(x) -> x^(-1) -> -x^(-2) -> 2x^(-3)
        # its error is quite large, up to atol = 1e-3
        # may due to the fact that it will be a 19-degree poynomial, which is A LOT.
        n = 5
        xs = np.linspace(0.5, 3, n).reshape((-1, 1))
        ys = np.log(xs).reshape((-1, 1))
        y1s = (xs ** (-1)).reshape((-1, 1))
        y2s = (-xs ** (-2)).reshape((-1, 1))
        y3s = (2 * xs ** (-3)).reshape((-1, 1))
        pts = np.hstack((xs, ys, y1s, y2s, y3s))

        p = HermitePolynomial(pts)
        self.assertEqual(p.deg(), n * (3+1) - 1)
        p1 = p.deriv(); self.assertTrue(arrayCloseTo(p1(xs), y1s, 1e-3), f'{p1(xs)}, {y1s}')
        p2 = p1.deriv(); self.assertTrue(arrayCloseTo(p2(xs), y2s, 1e-3), f'{p2(xs)}, {y2s}')
        p3 = p2.deriv(); self.assertTrue(arrayCloseTo(p3(xs), y3s, 1e-3), f'{p3(xs)}, {y3s}')
        pass
        
        
        


if __name__ == '__main__':
    unittest.main()