class Value:
    # stores a single value
    def __init__(self, n, parents = (), op = lambda x, y: x):
        self.val = float(n)
        self.grad = 0
        self.parents = parents # each parent is a value
        self.op = op # what op is used by the parents?

    def __str__(self):
        return str(self.val)
    
    def __repr__(self):
        return self.val
    
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.val + other.val, (self, other), lambda x, y: x + y)
    
    def __radd__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(other.val + self.val, (other, self), lambda x, y: x + y)    
    
    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.val - other.val, (self, other), lambda x, y: x - y)
    
    def __rsub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(other.val - self.val, (other, self), lambda x, y: x - y)    
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.val * other.val, (self, other), lambda x, y: x * y)
    
    def __rmul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(other.val * self.val, (other, self), lambda x, y: x * y)    

    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.val / other.val, (self, other), lambda x, y: x / y)
    
    def __rtruediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)        
        return Value(other.val / self.val, (other, self), lambda x, y: x / y)
    
    def __pow__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.val ** other.val, (self, other), lambda x, y: x ** y)
    
    def __rpow__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(other.val ** self.val, (other, self), lambda x, y: x ** y)
    
    def backwards(self):
        if self.grad == 0:
            self.grad = 1
        if self.parents == ():
            return
        self.parents[0].grad = self.grad * self.derivative(self.parents[0].val, self.parents[1].val, self.op)
        self.parents[1].grad = self.grad * self.derivative(self.parents[1].val, self.parents[0].val, self.op)
        self.parents[0].backwards()
        self.parents[1].backwards()

    @staticmethod
    def derivative(op1, op2, func):
        """
        derivative is [f(x + h) - f(x)]/h
        """
        h = 0.00001
        initial_val = func(op1, op2)
        bumped_val = func(op1 + h, op2)
        res = (bumped_val - initial_val) / h
        return res


g = Value(4)
h = Value(2)
g *= h
g.backwards()
print(h.grad)
