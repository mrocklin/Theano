from theano.gof.match import construct, deconstruct, rule
from theano.printing import debugprint
from theano import tensor as T

from logpy import isvar, var

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

    assert var(x) in deconstruct(2*x, (x,))[1:]

def test_match():
    rl = rule(x + x, 2*x, (x,))  # in, out, wilds
    assert streq(next(rl(y + y)), 2*y)
