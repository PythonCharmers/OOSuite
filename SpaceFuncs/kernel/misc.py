
class SpaceFuncsException(BaseException):
    def __init__(self,  msg):
        self.msg = msg
    def __str__(self):
        return self.msg

pwSet = set()
def pWarn(msg):
    if msg in pwSet: return
    pwSet.add(msg)
    print('SpaceFuncs warning: ' + msg)
    
def SF_error(msg):
    raise SpaceFuncsException(msg)
