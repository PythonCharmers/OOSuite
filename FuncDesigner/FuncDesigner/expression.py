lvl1_ops = ('+', '-')
lvl2_ops = ('*', )
lvl3_ops = ('/', )
lvl4_ops = ('**', )
ops = lvl1_ops + lvl2_ops + lvl3_ops + lvl4_ops

def getitem_expression(oof, ind, *args, **kw):
    r = oof.expression()
    if any(op in r for op in ops):
        r = '(' + r + ')'
    if type(ind) == slice:
        Ind = str(ind.start)+':'+str(ind.stop)
        if ind.step is not None:
            Ind += ':'+str(ind.step)
    else:
        Ind = str(ind)
    return r + '[' + Ind + ']'

def add_expression(Self, Other, *args, **kw):
    from ooFun import oofun
    tmp2 = (Other.expression(**kw) if isinstance (Other, oofun) else str(Other))
    r = Self.expression(**kw) + (' - ' + tmp2[1:] if tmp2[0] == '-' else ' + ' + tmp2)
    return r

def mul_expression(Self, Other, *args, **kw):
    from ooFun import oofun
    isOOFun = isinstance(Other, oofun)
    r1 = Self.expression(**kw)
    needBrackets1 = '+' in r1 or '-' in r1# or '*' in r1 or '/' in r1
    R1 = '(' + r1 + ')' if needBrackets1 else r1

    r2 = Other.expression(**kw) if isOOFun else str(Other)
    needBrackets2 = '+' in r2 or '-' in r2 #or '*' in r2 or '/' in r2
    R2 = '(' + r2 + ')' if needBrackets2 else r2
    r = (R1 + '*' + R2) if isOOFun else (R2 + '*' + R1)
    return r
