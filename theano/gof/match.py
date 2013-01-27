from logpy import var, isvar, run, eq, fact
from logpy.assoccomm import eq_assoccomm as eqac
from logpy.assoccomm import commutative, associative
import itertools as it


def deconstruct(x, variables=()):
    if x in variables:
        return var(x)
    elif x.owner is None:
        return x
    else:
        return (x.owner.op,) + tuple(deconstruct(i, variables)
                                            for i in x.owner.inputs)

def construct(tup):
    if isvar(tup):
        return tup.token
    elif not isinstance(tup, tuple):
        return tup
    else:
        return tup[0](*map(construct, tup[1:]))

def rule(in_pattern, out_pattern, variables):
    decon = lambda x: deconstruct(x, variables)
    return lambda expr: it.imap(construct,
            run(None,
                deconstruct(out_pattern, variables),
                eqac(deconstruct(in_pattern, variables), deconstruct(expr))))

from theano.tensor import Elemwise
from theano.scalar.basic import mul, add

fact(commutative, Elemwise(mul))
fact(commutative, Elemwise(add))
fact(associative, Elemwise(mul))
fact(associative, Elemwise(add))
