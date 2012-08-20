from baseProblem import MatrixProblem
import numpy as np

class TSP(MatrixProblem):
    _optionalData = []
    probType = 'TSP'
    expectedArgs = ['graph', 'objective']
    allowedGoals = ['travelling salesman path']
    showGoal = False
    allowRevisit = False
    _init = False

    # !!!!!!!!!!!!!! TODO: handle it
    startNode = None 
    
    def __setattr__(self, attr, val): 
        if self._init: self.err('openopt TSP instances are immutable, arguments should pass to constructor or solve()')
        self.__dict__[attr] = val
    
    def __init__(self, *args, **kw):
        self.objective = 'weight'
        MatrixProblem.__init__(self, *args, **kw)
        self.__init_kwargs = kw
        self._init = True


    def solve(self, *args, **kw):
        # DEBUG
        #from time import time
        #T = time()
        
        if len(args) > 1:
            self.err('''
            incorrect number of arguments for solve(), 
            must be at least 1 (solver), other must be keyword arguments''')
        solver = args[0] if len(args) != 0 else kw.get('solver', self.solver)
        KW = self.__init_kwargs.copy()
        KW.update(kw)
        objective = KW.get('objective', self.objective)
        if isinstance(objective, (list, tuple, set)):
            nCriteria = len(self.objective)
            if 3 * nCriteria != np.asarray(self.objective).size:
                objective = [(objective[3*i], objective[3*i+1], objective[3*i+2]) for i in range(int(round(np.asarray(self.objective).size / 3)))]
            if len(objective) == 1:
                KW['fTol'], KW['goal'] = objective[0][1:]
        else:
            objective = [(self.objective, KW.get('fTol', getattr(self, 'fTol')), KW.get('goal', getattr(self, 'goal')))]
        nCriteria = len(objective)
        isMOP = nCriteria > 1
        mainCr = objective[0][0]
         
        import FuncDesigner as fd, openopt as oo 
        solverName = solver if type(solver) == str else solver.__name__
        is_interalg = solverName == 'interalg'
        is_interalg_raw_mode = is_interalg and KW.get('dataHandling', oo.oosolver(solver).dataHandling) in ('auto','raw')
        P = oo.MOP if nCriteria > 1 else oo.GLP if is_interalg else oo.MILP

        import networkx as nx
        graph = self.graph # must be networkx Graph instance
        
        init_graph_is_directed = graph.is_directed()
        init_graph_is_multigraph = graph.is_multigraph()
        if not init_graph_is_multigraph or not init_graph_is_directed:
            graph = nx.MultiDiGraph(graph) #if init_graph_is_directed else nx.MultiGraph(graph)
        
        nodes = graph.nodes()
        edges = graph.edges()
        n = len(nodes)
        m = len(edges)

        node2index = dict([(node, i) for i, node in enumerate(nodes)])
        
        # TODO: implement MOP with new mode (requires interpolation interval analysis for non-monotone funcs)
        new = 1
        #is_interalg_raw_mode = False
        if not is_interalg or isMOP:
            # !!!!!!!!!!!!! TODO: add handling of MOP with new?
            new = 0 
            
        if new:
            x = []
            edge_ind2x_ind_val = {}
        else:
            pass
            #x = fd.oovars(m, domain=bool)
            
        #u0 = fd.oovar()
        u = fd.hstack((1, fd.oovars(n-1, lb=2 if 1 or not is_interalg_raw_mode else 2.0/n, ub=n if 1 or not is_interalg_raw_mode else 1.0 )))# if not is_interalg_raw_mode else 2.0 + 0.005 / n
        if is_interalg_raw_mode:
            for i in range(n-1):
                u[1+i].domain = np.arange(2, n+1)
        for i in range(1, u.size):
            u[i]('u' + str(i))
        cr_values = dict([(obj[0], []) for obj in objective])
        constraints = []
        EdgesDescriptors, EdgesCoords = [], []
        # mb rework it by successors etc?
        
        isMainCrMin = objective[0][2] in ('min', 'minimum')
        
        for node in nodes:
            out_nodes = graph[node].keys()
            if len(out_nodes) == 0:
                self.err('input graph has node %s that does not lead to any other node; solution is impossible' % node)            
            Edges = graph[node]
            
            if init_graph_is_multigraph and not isMOP:
                W = {}
                for out_node in out_nodes:
                    ww = list(Edges[out_node].values())
                    for w in ww:
                        tmp = W.get(out_node, None)
                        if tmp is None:
                            W[out_node] = w
                            continue
                        th = tmp[mainCr]
                        w_main_cr_val = w[mainCr]
                        if isMainCrMin  == (th > w_main_cr_val):
                            W[out_node] = w
                Out_nodes, W = np.array(list(W.keys())), np.array(list(W.values()))
            else:
                W = np.hstack([list(Edges[out_node].values()) for out_node in out_nodes])
                Out_nodes = np.hstack([[out_node] * len(Edges[out_node]) for out_node in out_nodes])

            rr = np.array([w[mainCr] for w in W])
            if isMainCrMin:
                rr = -rr
            else:
                if objective[0][2] not in ('max', 'maximum'):
                    self.err('unimplemented for fixed value goal in TSP yet, only min/max is possible for now')
                    
            ind = rr.argsort()
            W = W[ind]
            out_nodes = Out_nodes[ind]

            lc = 0
            for i, w in enumerate(W):
                if new:
                    edge_ind2x_ind_val[len(EdgesCoords)] = (len(x), lc)
                lc += 1
                EdgesCoords.append((node, out_nodes[i]))
                EdgesDescriptors.append(w)

                for key, val in w.items():
                    #if node2index[key] < node2index[out_node]: continue
                    if key in cr_values:
                        cr_values[key].append(val)
            if new:
                x.append(fd.oovar(domain = np.arange(lc)))            
        
        m = len(EdgesCoords) # new value
        
        if new: 
            assert len(x) == n
            x = fd.ooarray(x)
        else:
            x = fd.oovars(m, domain=bool) # new m value
        for i in range(x.size):
            x[i]('x'+str(i))
#        if init_graph_is_directed:
        dictFrom = dict([(node, []) for node in nodes])
        dictTo = dict([(node, []) for node in nodes])
        for i, edge in enumerate(EdgesCoords):
            From, To = edge
            dictFrom[From].append(i)
            dictTo[To].append(i)
        
        engine = fd.XOR
        
        # number of outcoming edges = 1
        if not new:
            for node, edges_inds in dictFrom.items():
                # !!!!!!!!!! TODO for interalg_raw_mode: and if all edges have sign similar to goal
                if 1 and is_interalg_raw_mode:
                    c = engine([x[j] for j in edges_inds])
                else:
                    nEdges = fd.sum([x[j] for j in edges_inds]) 
                    c =  nEdges >= 1 if self.allowRevisit else nEdges == 1
                constraints.append(c)

        # number of incoming edges = 1
        for node, edges_inds in dictTo.items():
            if len(edges_inds) == 0:
                self.err('input graph has node %s that has no edge from any other node; solution is impossible' % node)
            
            if new:
                x_inds, x_vals = [], []
                for elem in edges_inds:
                    x_ind, x_val = edge_ind2x_ind_val[elem]
                    x_inds.append(x_ind)
                    x_vals.append(x_val)
                c = engine([(x[x_ind] == x_val)(tol = 0.5) for x_ind, x_val in zip(x_inds, x_vals)])
            else:
                if 0 and is_interalg_raw_mode:
                    c = engine([x[j] for j in edges_inds])
                else:            
                    nEdges = fd.sum([x[j] for j in edges_inds]) 
                    c =  nEdges >= 1 if self.allowRevisit else nEdges == 1
            constraints.append(c)
        
        # MTZ
        for i, edge in enumerate(EdgesCoords):
            I, J = edge
            ii, jj = node2index[I], node2index[J]
            if ii != 0 and jj != 0:
                if new:
                    x_ind, x_val = edge_ind2x_ind_val[i]
                    c = fd.ifThen((x[x_ind] == x_val)(tol=0.5), u[ii] - u[jj]  <= - 1.0)
                elif is_interalg_raw_mode:
                    c = fd.ifThen(x[i], u[ii] - u[jj]  <= - 1.0)#u[jj] - u[ii]  >= 1)
                else:
                    c = u[ii] - u[jj] + 1 <= (n-1) * (1-x[i])
                constraints.append(c)
        
        # handling objective(s)
        FF = []
        
        for obj in objective:
            optCrName = obj[0]
            tmp = cr_values[optCrName]

            if len(tmp) == 0:
                self.err('seems like graph edgs have no attribute "%s" to perform optimization on it' % optCrName)
            elif len(tmp) != m:
                self.err('for optimization creterion "%s" at least one edge has no this attribute' % optCrName)
            if new:
                F = []
                lc = 0
                for X in x:
                    #domain = X.aux_domain
                    domain = X.domain
                    vals = [tmp[i] for i in range(lc, lc + domain.size)]
                    lc += domain.size
                    #F = sum(x)
                    F.append(fd.interpolator(domain, vals, k=1, s=0.00000001)(X))
                F = fd.sum(F)
            else:
                F = fd.sum(x*tmp)
            FF.append((F, obj[1], obj[2]))
        
        
        startPoint = {x:[0]*(m if not new else n)}#, u0:1}
        startPoint.update(dict([(U, n) for U in u[1:]]))

        p = P(FF if isMOP else F, startPoint, constraints = constraints)#, fixedVars = fixedVars)
        
        #print ('init Time in solve(): %0.1f' % (time()-T))
        KW.pop('objective', None)
        r = p.solve(solver, **KW)

        if P != oo.MOP:
            if new:
                x_ind_val2edge_ind = dict([(elem[1], elem[0]) for elem in edge_ind2x_ind_val.items()])
                SolutionEdges = [(EdgesCoords[i][0], EdgesCoords[i][1], EdgesDescriptors[i]) for i in [x_ind_val2edge_ind[(ind, x[ind](r))] for ind in range(n)]]
            else:
                SolutionEdges = [(EdgesCoords[i][0], EdgesCoords[i][1], EdgesDescriptors[i]) for i in range(m) if r.xf[x[i]] == 1]
            SE = [SolutionEdges[0]]
            for i in range(len(SolutionEdges)-1):
                SE.append(SolutionEdges[SE[-1][1]])
            SolutionEdgesCoords = [(elem[0], elem[1]) for elem in SE]

            r.nodes = [nodes[0]] + [edge[1] for edge in SolutionEdgesCoords]
            r.edges = SolutionEdgesCoords
            r.Edges = SE
            r.ff = p.ff
        else:
            r.solution = 'for MOP see r.solutions instead of r.solution'
            tmp_c, tmp_v = r.solutions.coords, r.solutions.values
            r.solutions = MOPsolutions([[(EdgesCoords[i][0], EdgesCoords[i][1], EdgesDescriptors[i]) for i in range(m) if Point[x[i]] == 1] for Point in r.solutions])
            r.solutions.values = tmp_v
        return r, p

#    def objFunc(self, x):
#        return dot(self.f, x) + self._c
class MOPsolutions(list):
    pass
