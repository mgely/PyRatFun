from os.path import join,dirname
import sys
sys.path.append(join(join(dirname(dirname(__file__)),'src')))
import core
import unittest
from math import isclose
import numpy as np


def cutoff_digits(f,digits):
    float_format = '%%.%de'%digits
    return float(float_format%(np.real(f))) + 1j*float(float_format%(np.imag(f)))

class TestCaseAppended(unittest.TestCase):

    def assertRelativelyClose(self,a,b,digits = 6):
        a = cutoff_digits(a,digits)
        b = cutoff_digits(b,digits)
        self.assertEqual(a,b)

    def assertArrayRelativelyClose(self,a,b,digits = 6):
        a = np.array(a)
        b = np.array(b)
        self.assertTrue(a.shape==b.shape,msg = f'Arrays do not have the same dimension {a.shape}!={b.shape}')
        for index,_ in np.ndenumerate(a):
            a_comp = cutoff_digits(a[index],digits)
            b_comp = cutoff_digits(b[index],digits)
            self.assertEqual(a_comp,b_comp,
                    msg = f'Components with index {index} do not match {a_comp}!={b_comp}')

class Polynomial(TestCaseAppended):

    def test_init(self):
        for coef in [
            1e-12,
            0,
            0.,
            -1,
            1j,
            [1,-2+2j,3j],
            core.Polynomial([1,-2+2j,3j]),
        ]:
            core.Polynomial(coef)
        
    def test_roots(self):
        for u in [
            [1,2],
            [3,5,7],
            [3,5,7,9],
            [3,5,7,9,11],
            [3,5j+1,7],
            [3,5j+1,7e-15],
            ]:
            self.assertArrayRelativelyClose(
                core.Polynomial(u).roots_companion(),
                core.Polynomial(u).roots_laguerre())

    def test_gcd(self):
        for u,v,g in [
            [[6,7,1],[-6,-5,1],[1,1]],
            ]:
            self.assertEqual(
                core.gcd(
                    core.Polynomial(u),
                    core.Polynomial(v)),
                    core.Polynomial(g))

class RationalFunction(TestCaseAppended):

    def test_init(self):
        for u,v in [
            [[1],[1]],
            [1,[1]],
            [[1],1],
            [[1,1],[1]],
            [[1j+1],[-1j,3]],
            [[3,3,3],[1,1]]
            ]:
            core.RationalFunction(u,v)

    def test_eq(self):
        u,v = np.array([3,-1+1j]),np.array([2,1+1j])
        self.assertEqual(
            core.RationalFunction(u,v),
            core.RationalFunction(u,v))
        self.assertEqual(
            core.RationalFunction(u,v),
            core.RationalFunction(-3*u,-3*v))

    def test_ne(self):
        u,v = [3,-1+1j],[2,1+2j]
        v_dif = [2,1+1j]
        self.assertNotEqual(
            core.RationalFunction(u,v),
            core.RationalFunction(u,v_dif))
        self.assertNotEqual(
            core.RationalFunction(u,v),
            core.RationalFunction(v,u))
            
    def test_pos(self):
        u,v = [3,-1+1j],[2,1+2j]
        self.assertEqual(
            +core.RationalFunction(u,v),
            core.RationalFunction(u,v))
            
    def test_neg(self):
        u,v = np.array([3,-1+1j]),np.array([2,1+2j])
        self.assertEqual(
            -core.RationalFunction(u,v),
            core.RationalFunction(-u,v))
        self.assertEqual(
            -core.RationalFunction(u,v),
            core.RationalFunction(u,-v))

    def test_add_sub_mul_div(self):
        v = core.Polynomial([1,1])
        u1 = core.Polynomial([3,5,1])
        u2 = core.Polynomial([1j-1,32e-15,1])
        self.assertEqual(
            core.RationalFunction(u1,v)+core.RationalFunction(u2,v),
            core.RationalFunction(u1+u2,v)
        )
        self.assertEqual(
            core.RationalFunction(u1,v)-core.RationalFunction(u2,v),
            core.RationalFunction(u1-u2,v)
        )
        self.assertEqual(
            core.RationalFunction(u1,v)*core.RationalFunction(u2,v),
            core.RationalFunction(u1*u2,v*v)
        )
        self.assertEqual(
            core.RationalFunction(u1,v)/core.RationalFunction(u2,v),
            core.RationalFunction(u1,u2)
        )

        u1,v1 = [1,1],[2,1]
        u2,v2 = [7,1],[11,1]
        u3,v3 = [25,21,2],[22,13,1]
        self.assertEqual(
            core.RationalFunction(u1,v1)+core.RationalFunction(u2,v2),
            core.RationalFunction(u3,v3)
        )
        u3,v3 = [-3,3],[22,13,1]
        self.assertEqual(
            core.RationalFunction(u1,v1)-core.RationalFunction(u2,v2),
            core.RationalFunction(u3,v3)
        )
        u3,v3 = [7,8,1],[22,13,1]
        self.assertEqual(
            core.RationalFunction(u1,v1)*core.RationalFunction(u2,v2),
            core.RationalFunction(u3,v3)
        )
        u3,v3 = [11,12,1],[14,9,1]
        self.assertEqual(
            core.RationalFunction(u1,v1)/core.RationalFunction(u2,v2),
            core.RationalFunction(u3,v3)
        )

    def test_pow(self):
        u,v = np.array([-1+1j,1]),np.array([1+2j,1])
        r = core.RationalFunction(u,v)
        self.assertEqual(
            r*r,
            r**2)
        self.assertEqual(
            r*r*r,
            r**3)
        self.assertEqual(
            r*r*r*r,
            r**4)
        
    def test_call(self):
        u = [-2+6j,-7+9j,-3+3j]
        v = [4-4j,22+2j,6+24j]
        self.assertRelativelyClose(
            core.RationalFunction(u,v)(5+13j),
            77855/651946 + (193209j)/1303892
        )
    
    def test_repr_str(self):
        u = [-2+6j,-7+9j,-3+3j]
        v = [4-4j,22+2j,6+24j]
        str(core.RationalFunction(u,v))
        repr(core.RationalFunction(u,v))

    def test_deriv(self):
        u,v = [7,1],[11,1]
        self.assertEqual(
            core.RationalFunction(u,v).deriv(),
            core.RationalFunction([4],[11**2,22,1])
        )

if __name__ == "__main__":
    unittest.main()