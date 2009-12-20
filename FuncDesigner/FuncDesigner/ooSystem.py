

class ooSystem(set):
    __call__ = lambda self, point: ooSystemState([(elem, elem(point)) for elem in self])
        
class ooSystemState(dict):
    # TODO: add .pp (prettyprint)
    
    def __repr__(self):
        r = []
        for key, val in self.items():
            r.append(str(key)+'='+str(val)+'\n')
        return ''.join(r)[:-1]


#def EmptyFunc(*args, **kwargs):
#    print ('warning - while invoking ooSystemState.pp (pretty print of ooSystemState instance) you could omit brackets')















