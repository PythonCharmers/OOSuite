class FuncDesignerException:
    def __init__(self,  msg):
        self.msg = msg
    def __str__(self):
        return self.msg

def checkSizes(a, b):
    if a.size != 1 and b.size != 1 and a.size != b.size:
        raise FuncDesignerException('operation of oovar/oofun ' + a.name + \
        ' and object with inappropriate size:' + str(a.size) + ' vs ' + b.size)
