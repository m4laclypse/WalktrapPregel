from pregel import Vertex, Pregel


import networkx as nx
import copy
import numpy as np
import time
from heapq import heappush, heappop, merge
import matplotlib.pyplot as plt
import pathlib
import os.path
import csv


class RandomWalkVertex(Vertex):
    def update(self):
        if self.superstep == 0:
            self.outgoing_messages = [(vertex, self.value) for vertex in self.out_vertices]

        elif self.superstep < self.t:
            P_xt = np.zeros(self.num_vertices)
            for (vertex, P) in self.incoming_messages:
                P_xt += P
            self.outgoing_messages = [(vertex, P_xt) for vertex in self.out_vertices]
            self.value = P_xt
        else:
                self.active = False

def randomWalk(G, A, t):
    N = A.shape[0]
    vertices = [0] * N
    for i in range(N):
        vertex = RandomWalkVertex(i, 0, [])
        vertex.t = t
        vertex.num_vertices = N
        vertices[i] = vertex
    vertices = np.array(vertices)
    for i in range(N):
        A_i = A[i]
        for j in range(N):
            if A_i[j] == 1:
                vertices[i].out_vertices.append(vertices[j])
        vertices[i].value = A_i
    
    p = Pregel(vertices, 8)
    p.run()
    return np.array([vertex.value for vertex in p.vertices])


def walktrap(G, t, tRW):

    for vertex in G.nodes:
        G.add_edge(vertex, vertex)

    G = nx.convert_node_labels_to_integers(G)
    N = G.number_of_nodes()

    A = np.array(nx.to_numpy_matrix(G))

    Dx = np.zeros((N,N))
    P = np.zeros((N,N))
    for i, A_row in enumerate(A):
        d_i = np.sum(A_row)
        P[i] = A_row / d_i
        Dx[i,i] = d_i ** (-0.5)

    P_t = randomWalk(G, A, tRW)

    # Weight of all the edges excluding self-edges
    G_total_weight = G.number_of_edges() - N


    class RandomWalkVertex(Vertex):
        def modularity(self):
            return (self.internal_weight - (self.total_weight*self.total_weight/G_total_weight)) / G_total_weight

        def custom_init(self, id, t=200):
            self.id = id
            self.community = str(id)
            self.communityMembers = set([])
            self.history = [str(id)]
            self.internal_weight = 0
            self.total_weight = self.internal_weight + (len([id for id, edge in enumerate(A[self.id]) if edge == 1 and id != self.id])/2)
            self.vert = set([id])
            self.P_c = P_t[self.id]
            self.size = 1
            self.min_sigma_heap = []
            self.t = t
            self.neighbourCommu = {}
            self.minDeltaSigma = None
            self.defunctCommunities = set([])
            self.modularities = [self.modularity()]
            self.events = [0]
            self.sentFusion = False

        def update(self):
            if self.superstep == 0:
                self.outgoing_messages = [(vertex, ("delta", self.community, self.min_sigma_heap[0], self.communityMembers)) for vertex in set(self.out_vertices + list(self.communityMembers))]

            elif self.superstep < self.t:
                self.min_sigma_heap.sort()


                types = [x[1][0] for x in self.incoming_messages]

                 # if "initFusion" in types and self.sentFusion:
                 #   numMessage = types.index("initFusion")
                 #   out_message = list(self.incoming_messages[numMessage][1])
                 #   out_message[0] = "fusion"
                 #   out_message = tuple(out_message)
                 #   self.outgoing_messages = [(vertex, out_message) for vertex in self.communityMembers] + [(self, out_message)]

                if "fusion" in types:
                    self.sentFusion = False
                    numMessage = types.index("fusion")
                    _, otherId, otherCommu, otherSize, otherP_c, otherInternal_weight, otherTotal_weight, otherVert, otherNeighbourCommu, deltaSigma, otherCommunityMembers, otherMinSigmaHeap, otherDefunct = self.incoming_messages[numMessage][1]
                    oldSize = self.size
                    oldCommu = self.community
                    self.defunctCommunities = self.defunctCommunities.union(otherDefunct)
                    self.defunctCommunities.add(self.community)
                    self.defunctCommunities.add(otherCommu)
                    self.communityMembers = self.communityMembers.union(otherCommunityMembers)
                    self.communityMembers.add(self.incoming_messages[numMessage][0])
                    self.community = (min(self.community, otherCommu) + "_" + max(self.community, otherCommu))
                    self.history.append(self.community)
                    self.size = self.size + otherSize
                    self.P_c = (oldSize * self.P_c + otherSize * otherP_c) / self.size
                    oldVert = self.vert
                    self.vert = self.vert.union(otherVert)
                    two_commu_weight = 0
                    for v1 in oldVert:
                        for id, edge in enumerate(A[v1]):
                            if edge == 1 and id in otherVert:
                                two_commu_weight += 1
                    self.internal_weight = self.internal_weight + otherInternal_weight + two_commu_weight
                    self.total_weight = self.total_weight + otherTotal_weight
                    oldNeighbourCommu = self.neighbourCommu
                    self.neighbourCommu = {**self.neighbourCommu, **otherNeighbourCommu}
                    self.min_sigma_heap = list(merge(self.min_sigma_heap, otherMinSigmaHeap))

                    self.events.append(self.superstep)
                    self.modularities.append(self.modularity())

                    self.outgoing_messages = []
                    deltaS = heappop(self.min_sigma_heap)[0]
                    for C_id in [x for x in self.neighbourCommu]:
                        if C_id.community != self.community:
                            # If C is neighbour of both C1 and C2 then we can apply Theorem 4
                            if (C_id in [x for x in oldNeighbourCommu]) and (C_id in [x for x in otherNeighbourCommu]):
                                infoC1C = oldNeighbourCommu[C_id]
                                infoC2C = otherNeighbourCommu[C_id]
                                delta_sigma_C1C = infoC1C[0]
                                delta_sigma_C2C = infoC2C[0]
                                ds = (( (oldSize + int(infoC2C[1]))*(delta_sigma_C1C) / (self.size + int(infoC2C[1])) + ((otherSize + int(infoC2C[1]))*(delta_sigma_C2C) - (int(infoC2C[1])*deltaS)) / (self.size + int(infoC2C[1]))))
                                self.neighbourCommu[C_id] = (ds, self.community, C_id.community)

                                delta_sigma = (ds, min(self.community, C_id.community), max(self.community, C_id.community))
                                if delta_sigma not in self.min_sigma_heap:
                                    heappush(self.min_sigma_heap, delta_sigma)

                            # Otherwise apply Theorem 3 to (C, C3)
                            else:
                                ds = np.sum(np.square( np.matmul(Dx, C_id.P_c) - np.matmul(Dx, self.P_c) )) * C_id.size*self.size / ((C_id.size + self.size) * N)

                                delta_sigma = (ds, min(self.community, C_id.community), max(self.community, C_id.community))
                                self.neighbourCommu[C_id] = delta_sigma
                                if delta_sigma not in self.min_sigma_heap:
                                    heappush(self.min_sigma_heap, delta_sigma)

                            self.outgoing_messages.append((C_id, ("synchroF", self.community, (ds, self.community, C_id.community), self.size, self.P_c)))
                            self.outgoing_messages.append((C_id, ("defunct", oldCommu)))
                            self.outgoing_messages.append((C_id, ("defunct", otherCommu)))

                    for ds in self.min_sigma_heap:
                        if ds[1] == oldCommu or ds[2] == otherCommu:
                            self.min_sigma_heap.remove(ds)
                            self.minDeltaSigma = None
                        elif ds[1] == ds[2]:
                            self.min_sigma_heap.remove(ds)
                            self.minDeltaSigma = None       
                    
                else:
                    self.outgoing_messages = []
                    hasSynchro = False
                    deltaSigmaChanged = False

                    for (vertex, message) in [x for x in self.incoming_messages if x[1][0] == "defunct"]:
                        for ds in self.min_sigma_heap:
                            if ds[1] == message[1] or ds[2] == message[1]:
                                self.min_sigma_heap.remove(ds)
                                self.minDeltaSigma = None
                        self.defunctCommunities.add(message[1])

                    for (vertex, message) in [x for x in self.incoming_messages if x[1][0][:7] == "synchro"]:
                        if message[0] == "synchroF":
                            ds = message[2]
                            if ds[1] in self.defunctCommunities:
                                self.outgoing_messages.append((vertex, ("defunct", ds[1])))
                            elif ds[2] in self.defunctCommunities:
                                self.outgoing_messages.append((vertex, ("defunct", ds[2])))

                            else:
                                if ds not in self.min_sigma_heap:
                                    heappush(self.min_sigma_heap, ds)
                                    hasSynchro = True
                                for member in self.communityMembers:
                                    self.outgoing_messages.append((member, ("synchro", message[1], message[2], message[3], message[4])))

                        if message[0] == "synchro":
                            ds = message[2]
                            if ds[1] in self.defunctCommunities:
                                self.outgoing_messages.append((vertex, ("defunct", ds[1])))
                            elif ds[2] in self.defunctCommunities:
                                self.outgoing_messages.append((vertex, ("defunct", ds[2])))
                            else:
                                if ds not in self.min_sigma_heap:
                                    heappush(self.min_sigma_heap, ds)
                                    hasSynchro = True

                    for (vertex, message) in [x for x in self.incoming_messages if x[1][0] == "delta"]:
                        ds = message[2]
                        if ds[1] in self.defunctCommunities:
                            self.outgoing_messages.append((vertex, ("defunct", ds[1])))
                        elif ds[2] in self.defunctCommunities:
                            self.outgoing_messages.append((vertex, ("defunct", ds[2])))
                        else:
                            if ds not in self.min_sigma_heap:
                                heappush(self.min_sigma_heap, ds)
                    
                    for (vertex, message) in [x for x in self.incoming_messages if x[1][0] == "hold"]:
                        deltaSigmaChanged = True
                        hasSynchro = True
                    
                    if self.min_sigma_heap != []:
                        try:
                            newMin = min(self.minDeltaSigma, self.min_sigma_heap[0])
                            deltaSigmaChanged = newMin != self.minDeltaSigma
                            self.minDeltaSigma = newMin
                        except (TypeError, IndexError) as e:
                            try:
                                newMin = self.min_sigma_heap[0]
                                deltaSigmaChanged = newMin != self.minDeltaSigma
                                self.minDeltaSigma = newMin
                            except IndexError:
                                deltaSigmaChanged = True
                    

                    if deltaSigmaChanged or hasSynchro:
                        self.outgoing_messages += [(vertex, "hold") for vertex in self.communityMembers]
                        
                    if self.superstep % 5 == 0 and self.min_sigma_heap != []:
                        self.outgoing_messages += [(vertex, ("delta", self.community, self.minDeltaSigma, self.communityMembers)) for vertex in set(self.out_vertices + list(self.communityMembers))]
                    
                    if self.superstep % 5 == 1 and deltaSigmaChanged and self.min_sigma_heap != []:
                        self.outgoing_messages += [(vertex, ("synchro", self.community, self.minDeltaSigma, self.communityMembers)) for vertex in set(self.out_vertices + list(self.communityMembers))]
                    
                    if self.min_sigma_heap != [] and not hasSynchro and self.superstep % 10 == 8 and self.community in self.minDeltaSigma[1:]:
                        if str(self.minDeltaSigma[1]) == str(self.community):
                            otherCommu = str(self.minDeltaSigma[2])
                        else:
                            otherCommu = str(self.minDeltaSigma[1])
                        out_message = (
                                "fusion", self.id, self.community, self.size, self.P_c, self.internal_weight, self.total_weight, self.vert, self.neighbourCommu, self.minDeltaSigma,
                                self.communityMembers, self.min_sigma_heap, self.defunctCommunities
                                )
                        
                        self.outgoing_messages += [(vertex, out_message) for vertex in self.allVertices if vertex.community == otherCommu]
                        self.sentFusion = True
            else:
                self.active = False
    
    vertices = [0] * N
    for i in range(N):
        vertex = RandomWalkVertex(i, 0, [])
        vertex.custom_init(i, t)
        vertices[i] = vertex
    vertices = np.array(vertices)
    for vertex in vertices:
        vertex.allVertices = vertices

    for i in range(N):
        A_i = A[i]
        for j in range(N):
            if A_i[j] == 1:
                vertices[i].out_vertices.append(vertices[j])
                if i != j:
                    ds = (0.5/N) * np.sum(np.square(np.matmul(Dx, P_t[i]) - np.matmul(Dx, P_t[j])))

                    delta_sigma = (ds, min(str(i), str(j)), max(str(i), str(j)))
                    if delta_sigma not in vertices[i].min_sigma_heap:
                        heappush(vertices[i].min_sigma_heap, delta_sigma)
                    vertices[i].neighbourCommu[vertices[j]] = delta_sigma
    
    
    p = Pregel(vertices, 8)
    p.run()

    dateEvents = []
    for vertex in vertices:
        dateEvents += vertex.events
    dateEvents = sorted(list(set(dateEvents)))

    modularities = []
    for event in dateEvents:
        temp = 0
        for vertex in vertices:
            # print(vertex.community, vertex.min_sigma_heap)
            try:
                index = next(i for i, v in enumerate(vertex.events) if v >= event)
                temp += vertex.modularities[index]
            except StopIteration:
                pass
        modularities.append(temp)

    print("Date des fusions : ", dateEvents)
    Qmax_index = np.argmax(modularities)
    print("On a un Q maximal après la fusion numéro : ", Qmax_index, " sur un total de ", len(dateEvents))
    timeMax = dateEvents[Qmax_index]

    partition = set([])
    dicCommunities = {}
    for vertex in vertices:
        try:
            index = next(i for i, v in enumerate(vertex.events) if v > timeMax) - 1
        except:
            index = len(vertex.events) - 1
        partition.add(vertex.history[index])
        if vertex.history[index] not in dicCommunities:
            dicCommunities[vertex.history[index]] = [vertex]
        else:
            dicCommunities[vertex.history[index]].append(vertex)

    return dicCommunities, partition, modularities


def couleurs(coms, N):
    couleurs = np.zeros(len(G.nodes))
    i = 0
    for key in coms:
        for vertex in coms[key]:
            couleurs[vertex.id] = i
        i += 1
    return couleurs


def walktrapNonPregel(G, t, add_self_edges=True, verbose=False):
    """
    Cette implémentation est celle des chercheurs à l'origine de l'article sur l'algorithme
    Walktrap, elle nous servira ici à comparer le temps d'exécution
    """
    ##################################################################
    class Community:
        def __init__(self, new_C_id, C1=None, C2=None):
            self.id = new_C_id
            # New community from single vertex
            if C1 is None:
                self.size = 1
                self.P_c = P_t[self.id] # probab vector
                self.adj_coms = {}
                self.vertices = set([self.id])
                self.internal_weight = 0. 
                self.total_weight = self.internal_weight + (len([id for id, x in enumerate(A[self.id]) if x == 1. and id != self.id])/2.) #External edges have 0.5 weight, ignore edge to itself
            # New community by merging 2 older ones
            else:
                self.size = C1.size + C2.size
                self.P_c = (C1.size * C1.P_c + C2.size * C2.P_c) / self.size
                # Merge info about adjacent communities, but remove C1, C2
                self.adj_coms = {**C1.adj_coms, **C2.adj_coms}
                del self.adj_coms[C1.id]
                del self.adj_coms[C2.id]
                self.vertices = C1.vertices.union(C2.vertices)
                weight_between_C1C2 = 0.
                for v1 in C1.vertices:
                    for id, x in enumerate(A[v1]):
                        if x == 1. and id in C2.vertices:
                            weight_between_C1C2 += 1.
                self.internal_weight = C1.internal_weight + C2.internal_weight + weight_between_C1C2
                self.total_weight = C1.total_weight + C2.total_weight

        def modularity(self):
            return (self.internal_weight - (self.total_weight*self.total_weight/G_total_weight)) / G_total_weight
    ##################################################################

    # If needed, add self-edges
    if add_self_edges:
        for v in G.nodes:
            G.add_edge(v, v)

    # G = nx.convert_node_labels_to_integers(G) # ensure that nodes are represented by integers starting from 0
    N = G.number_of_nodes()

    # Build adjacency matrix A
    A = np.array(nx.to_numpy_matrix(G))

    # Build transition matrix P from adjacency matrix
    # and diagonal degree matrix Dx of negative square roots degrees
    Dx = np.zeros((N,N))
    P = np.zeros((N,N))
    for i, A_row in enumerate(A):
        d_i = np.sum(A_row)
        P[i] = A_row / d_i
        Dx[i,i] = d_i ** (-0.5)

    # Take t steps of random walk
    P_t = np.linalg.matrix_power(P, t)

    # Weight of all the edges excluding self-edges
    G_total_weight = G.number_of_edges() - N

    # Total number of all communities created so far
    community_count = N
    # Dictionary of all communities created so far, indexed by comID
    communities = {}
    for C_id in range(N):
        communities[C_id] = Community(C_id)

    # Minheap to store delta sigmas between communitites: <deltaSigma(C1,C2), C1_id, C2_id>
    min_sigma_heap = []
    for e in G.edges:
        C1_id = e[0]
        C2_id = e[1]
        if C1_id != C2_id:
            # Apply Definition 1 and Theorem 3
            ds = (0.5/N) * np.sum(np.square( np.matmul(Dx,P_t[C1_id]) - np.matmul(Dx,P_t[C2_id]) ))
            heappush(min_sigma_heap, (ds, C1_id, C2_id))
            # Update each community with its adjacent communites
            communities[C1_id].adj_coms[C2_id] = ds
            communities[C2_id].adj_coms[C1_id] = ds  

    # Record delta sigmas of partitions merged at each step
    delta_sigmas = []
    # Store IDs of current communities for each k
    # Partitions is a list of length k that stores IDs of communities for each partitioning
    partitions = [] # at every step active communities are in the last entry of 'partitions'
    # Make first partition, single-vertex communities
    partitions.append(set(np.arange(N)))
    # Calculate modularity Q for this partition
    modularities = [np.sum([communities[C_id].modularity() for C_id in partitions[0]])]

    for k in range(1, N):
        # Current partition: partitions[k-1]
        # New partition to be created in this iteration: partitions[k]

        # Choose communities C1, C2 to merge, according to minimum delta sigma
        # Need to also check if C1_id and C2_id are communities at the current partition partitions[k-1]
        while not not min_sigma_heap:
            delta_sigma_C1C2, C1_id, C2_id = heappop(min_sigma_heap)
            if C1_id in partitions[k-1] and C2_id in partitions[k-1]:
                break
        # Record delta sigma at this step
        delta_sigmas.append(delta_sigma_C1C2)

        # Merge C1, C2 into C3, assign to it next possible ID, that is C3_ID = totComCnt
        C3_id = community_count
        community_count += 1 # increase for the next one
        communities[C3_id] = Community(C3_id, communities[C1_id], communities[C2_id])

        # Add new partition (k-th)
        partitions.append(copy.deepcopy(partitions[k-1]))
        partitions[k].add(C3_id) # add C3_ID
        partitions[k].remove(C1_id)
        partitions[k].remove(C2_id)

        # Update delta_sigma_heap with entries concerning community C3 and communities adjacent to C1, C2
        # Check all new neighbours of community C3
        for C_id in communities[C3_id].adj_coms.keys():
            # If C is neighbour of both C1 and C2 then we can apply Theorem 4
            if (C_id in communities[C1_id].adj_coms) and (C_id in communities[C2_id].adj_coms):
                delta_sigma_C1C = communities[C1_id].adj_coms[C_id]
                delta_sigma_C2C = communities[C2_id].adj_coms[C_id]
                # Apply Theorem 4 to (C, C3)
                ds = ( (communities[C1_id].size + communities[C_id].size)*delta_sigma_C1C + (communities[C2_id].size + communities[C_id].size)*delta_sigma_C2C - communities[C_id].size*delta_sigma_C1C2 ) / (communities[C3_id].size + communities[C_id].size)

            # Otherwise apply Theorem 3 to (C, C3)
            else:
                ds = np.sum(np.square( np.matmul(Dx,communities[C_id].P_c) - np.matmul(Dx,communities[C3_id].P_c) )) * communities[C_id].size*communities[C3_id].size / ((communities[C_id].size + communities[C3_id].size) * N)

            # Update min_sigma_heap and update delta sigmas between C3 and C
            heappush(min_sigma_heap, (ds , C3_id, C_id))
            communities[C3_id].adj_coms[C_id] = ds
            communities[C_id].adj_coms[C3_id] = ds  

        # Calculate and store modularity Q for this partition
        modularities.append(np.sum([communities[C_id].modularity() for C_id in partitions[k]]))

    return np.array(partitions), communities, np.array(delta_sigmas), np.array(modularities)


def importBitcoinAlpha():
    file1 = open(os.path.join(pathlib.Path(__file__).parent.absolute(), "soc-sign-bitcoinalpha.csv"), 'r')
    reader = csv.reader(file1)
    data = [(x[0], x[1]) for x in list(reader)]
    print(data)
    G = nx.Graph()
    G.add_edges_from(data)
    return G


def importBitcoinOtc():
    file1 = open(os.path.join(pathlib.Path(__file__).parent.absolute(), "soc-sign-bitcoinotc.csv"), 'r')
    reader = csv.reader(file1)
    data = [(x[0], x[1]) for x in list(reader)]
    G = nx.Graph()
    G.add_edges_from(data)
    return G


if __name__ == "__main__":
    print("Jeu de données Karaté")
    G = nx.karate_club_graph()
    pos = nx.spring_layout(G)

    t = time.time()
    coms, parts, Qs = walktrap(G, 600, 4)
    print("Le set Karaté nous a demandé ", time.time() - t, " secondes")
    coul = couleurs(coms, len(G.nodes))
    plt.figure(figsize=(11, 11))
    nx.draw(G, pos, node_color=coul)
    # plt.show()

    t = time.time()
    walktrapNonPregel(G, 4)
    print("Le set Karaté a demandé sans Pregel ", time.time() - t, " secondes")

    print("\n \n")

    print("Jeu de données BitcoinAlpha")
    G = importBitcoinAlpha()
    pos = nx.spring_layout(G)

    t = time.time()
    coms, parts, Qs = walktrap(G, 600, 4)
    print("Le set BitcoinAlpha nous a demandé ", time.time() - t, " secondes")
    coul = couleurs(coms, len(G.nodes))
    plt.figure(figsize=(11, 11))
    nx.draw(G, pos, node_color=coul)
    # plt.show()

    t = time.time()
    walktrapNonPregel(G, 4)
    print("Le set BitcoinAlpha a demandé sans Pregel ", time.time() - t, " secondes")

    print("\n \n")

    print("Jeu de données BitcoinOTC")
    G = importBitcoinOtc()
    pos = nx.spring_layout(G)

    t = time.time()
    coms, parts, Qs = walktrap(G, 600, 4)
    print("Le set BitcoinOTC nous a demandé ", time.time() - t, " secondes")
    coul = couleurs(coms, len(G.nodes))
    plt.figure(figsize=(11, 11))
    nx.draw(G, pos, node_color=coul)
    # plt.show()

    t = time.time()
    walktrapNonPregel(G, 4)
    print("Le set BitcoinOTC a demandé sans Pregel ", time.time() - t, " secondes")
