import theano
import logging
import numpy

logging.getLogger("theano.gof.cmodule").setLevel(logging.DEBUG)
theano.config.nocleanup = True

x = theano.tensor.matrix('x')
y = theano.tensor.dot(x,x)
mode = theano.Mode(optimizer=theano.compile.mode.get_default_mode().optimizer,
                    linker = theano.CLinker())
f = theano.function([x], y, mode=mode)
x = numpy.ones((5,5), dtype=numpy.float32)
f(x)
