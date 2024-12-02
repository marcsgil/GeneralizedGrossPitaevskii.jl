using SciMLBase

f(x) = x

@code_warntype isinplace(f, 2)