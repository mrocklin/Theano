from theano.gof.match import construct, deconstruct
from theano.printing import debugprint
from theano import tensor as T

from logpy import isvar

def streq(a, b):
    return debugprint(a, file='str') == debugprint(b, file='str')

x = T.matrix('x')
y = T.matrix('y')

def test_construct_deconstruct():
    expr = 2*x + y
    assert streq(expr, construct(deconstruct(expr)))
    assert isinstance(deconstruct(expr), tuple)
    assert deconstruct(x) == x

    assert isvar(deconstruct(x, variables=(x,)))
    assert streq(expr, construct(deconstruct(expr, (x,))))
