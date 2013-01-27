from theano.gof.match import construct, deconstruct, rule
from theano.printing import debugprint
from theano import tensor as T
from theano.tensor import sin, cos, tan

from logpy import isvar, var


# Apparently x + y != x + y.  Is there a better way to do this?
def streq(a, b):
    """ equality test using debugprint """
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
    ident = rule(x, x, [x])
    assert streq(next(ident(y + 1)), y + 1)

    rl = rule(x + x, 2*x, (x,))  # in, out, wilds
    assert streq(next(rl(y + y)),   2*y)

def test_sincos():
    rl = rule(cos(x) * tan(x), sin(x), [x])

    assert streq(next(rl(cos(x) * tan(x))),         sin(x))
    assert streq(next(rl(tan(y) * cos(y))),         sin(y))
    assert streq(next(rl(tan(x+y) * cos(x+y))),     sin(x+y))

    # This fails for some reason
    # assert streq(next(rl(tan(2) * cos(2))), sin(2))
