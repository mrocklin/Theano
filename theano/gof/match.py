from logpy import var, isvar

def deconstruct(x, variables=()):
    if x in variables:
        return var(x)
    elif x.owner is None:
        return x
    else:
        return (x.owner.op,) + tuple(map(deconstruct, x.owner.inputs))

def construct(tup):
    if isvar(tup):
        return tup.token
    elif not isinstance(tup, tuple):
        return tup
    else:
        return tup[0](*map(construct, tup[1:]))
