#Author: Balaji Sankeerth Jagini
#Mail Id: bjagini@albany.edu

import networkx as nx
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import numpy as np
from numpy import linalg as LA
import datetime as dt
from datetime import date
import calendar
import time
from operator import itemgetter


# Returns Directed graph with weighted edges and nodes as email address
def construct_Digraph():

    g = nx.DiGraph()


    with open("/Users/sankeerthjagini/Documents/Studies/NetSci/Project/532projectdataset.txt") as file:

        content = file.read().splitlines()
        numberofelems = len(content)



        for i in range(0, numberofelems):

            nodes = content[i].split()

            date1 = nodes[0]
            mydate = dt.datetime.fromtimestamp(int(date1))

            day = calendar.day_name[mydate.weekday()]

            if day != 'Saturday' and day != 'Sunday':

                extra = 0
                if g.has_edge(nodes[1], nodes[2]):
                    extra = g[nodes[1]][nodes[2]]['weight']

                g.add_edge(nodes[1], nodes[2], weight = 1 + extra)


            """
            if in_str == input_str:
                count = count + 1

            else:
                content1 = content[i-1].split()

                extra = 0
                if g.has_edge(content1[1], content1[2]):
                    extra = g[content1[1]][content1[2]]['weight']

                g.add_edge(content1[1], content1[2], weight=count+extra)
                a = content[i].split()
                in_str = a[1] + a[2]
                count = 1

            # eg.: last 10 elems are same or if only last elem is diff
            if i == numberofelems-1:
                content1 = content[i].split()

                extra = 0
                if g.has_edge(content1[1], content1[2]):
                    extra = g[content1[1]][content1[2]]['weight']

                g.add_edge(content1[1], content1[2], weight=count+extra)
            """

    return g




# Returns Directed graph with weighted edges and nodes as email address
def construct_graph():

    g = nx.Graph()
    with open("/Users/sankeerthjagini/Documents/Studies/NetSci/Project/532projectdataset.txt") as file:

        content = file.read().splitlines()
        numberofelems = len(content)

        for i in range(0, numberofelems):

            nodes = content[i].split()

            date1 = nodes[0]
            mydate = dt.datetime.fromtimestamp(int(date1))

            day = calendar.day_name[mydate.weekday()]

            if day != 'Saturday' and day != 'Sunday':

                extra = 0
                if g.has_edge(nodes[1], nodes[2]):
                    extra = g[nodes[1]][nodes[2]]['weight']

                g.add_edge(nodes[1], nodes[2], weight = 1 + extra)

    # print(g.edges())

    return g



# Return min,max,avg of indegree, outdegree, degree of nodes and also number of bidirectional edges
def min_max_avg(g):

    max_in=0
    max_out=0
    max_t=0
    n = len(g.nodes())
    a = n * (n-1)
    min_in = a
    min_out = a
    min_t = a
    ind = 0
    outd = 0
    totd = 0


    for i in g.nodes():
        indegree = g.in_degree(i)
        outdegree = g.out_degree(i)
        degree = g.degree(i)
        if indegree > max_in:
            max_in = indegree

        if outdegree > max_out:
            max_out = outdegree

        if degree > max_t:
            max_t = degree

        if indegree < min_in:
            min_in = indegree

        if outdegree < min_out:
            min_out = outdegree

        #Doubt
        if degree < min_t:
            min_t = degree

        ind = ind + indegree
        outd = outd + outdegree
        totd = totd + degree


    bidirectional(g)
    print("max_in: ", max_in)
    print("max_out: ", max_out)
    print("max_t: ", max_t)
    print("min_in: ", min_in)
    print("min_out: ", min_out)
    print("min_t: ", min_t)
    print("Average Number of in degree: ", ind/n)
    print("Average number of out degree: ", outd/n)
    print("Average number of total degree: ", totd/n)


# Prints number of bidirectional edges
def bidirectional(g):
    count=0
    for u,v in g.edges():
        if u in g[v]:
            count=count+1
    print("Number of Bidirectional edges: ", int(count/2))


def diameter(g):

    # z = nx.connected_component_subgraphs(g)
    # connected_diam = []
    # for i in z:
    #     p1 = nx.shortest_path_length(i)
    #     print("P1 calculated")
    #     max1 = 0
    #     print("num of nodes in subgraph: ",len(i.nodes()))
    #     count=0
    #
    #     for t in p1:
    #         print("node as key: ",t[0])
    #
    #
    #
    #     for j in p1:
    #
    #         if max1 < max(j[1].values()):
    #             print("values in dict: ",len(j[1].values()))
    #             max1 = max(j[1].values())
    #
    #         print(count)
    #         count = count +1
    #     print("max calculated")
    #     connected_diam.append(max1)
    #
    # # print(connected_diam)
    # return connected_diam

    z = nx.connected_component_subgraphs(g)
    connected_diam = []

    for i in z:

        max1 = 0
        nodes = list(i.nodes())
        num_nodes = len(i.nodes())

        # Calculating Geodesic distance
        dist = 0
        for l in range(0, num_nodes):
            for m in range(l, num_nodes):
                # print(nodes[l])
                # print(nodes[m])

                a = 0
                b = 0
                if nx.has_path(i, nodes[l], nodes[m]):
                    bb = (nx.shortest_path(i, nodes[l], nodes[m]))
                    a = len(nx.shortest_path(i, nodes[l], nodes[m])) - 1
                    # print(a, nodes[l], nodes[m])
                    # print(bb)
                if nx.has_path(i, nodes[m], nodes[l]):
                    b = len(nx.shortest_path(i, nodes[m], nodes[l])) - 1
                    # print(b, nodes[l], nodes[m])

                c = max(a, b)
                if max1 < c:
                    max1 = c

        print(max1)
        connected_diam.append(max1)

    return connected_diam


def plotting_helperfunc(degree_list):

    count_dictionary = dict()

    # Making a dictionary where key is degree of node and values are number of nodes with that degree
    for u in degree_list:
        if u[1] not in count_dictionary:
            count_dictionary[u[1]] = 1
        else:
            count_dictionary[u[1]] = count_dictionary[u[1]] + 1


    keys = []
    vals = []
    value = 0

    # This is  wrong we should count values not value of keys
    # value_list = list(count_dictionary.keys())
    value_list = list(count_dictionary.values())
    # In Degree Distribution y-axis is Probability i.e fraction of degree, so we need total value
    for vall in value_list:
        value = value+vall

    for key, val in count_dictionary.items():
        keys.append(key)
        vals.append(val/value)

    return keys,vals




def plotting(g):

    # Plotting on log-log scale
    plt.xscale('log')
    plt.yscale('log')


    # Plotting degree distribution and fitting corresponding least square regression line
    keys,vals = plotting_helperfunc(g.degree)
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Fraction of nodes')
    plt.plot(keys, vals, 'b.')


    # Returns coefficient of power-law line
    logkeys = []
    logvals = []

    rang =len(keys)
    for i in range(0,rang):
        if vals[i]>0:
            logkeys.append(np.math.log10(keys[i]))
            logvals.append(np.math.log10(vals[i]))

    # z = np.polyfit(v_u, e_u, 1)
    z = np.polyfit(logkeys, logvals, 1)
    # Exponent or slope of power-law
    m = z[0]
    a = z[1]

    y1 = []
    for i in keys:
        y1.append((10**a) * (i ** m))

    # print('Exponent power-law for eigen vs weight: ', z[0])
    plt.plot(keys, y1, 'b.', label=z[0], mfc='none')




    # Plotting in-degree distribution and fitting corresponding least square regression line
    keys1, vals1 = plotting_helperfunc(g.in_degree)
    plt.plot(keys1, vals1, 'g.')
    # Returns coefficient of power-law line
    logkeys1 = []
    logvals1 = []
    keys11=[]

    rang =len(keys1)
    for i in range(0,rang):
        if keys1[i]>0 and vals1[i]>0:
            logkeys1.append(np.math.log10(keys1[i]))
            logvals1.append(np.math.log10(vals1[i]))
            keys11.append(keys1[i])

    # z = np.polyfit(v_u, e_u, 1)
    z = np.polyfit(logkeys1, logvals1, 1)
    # Exponent or slope of power-law
    m = z[0]
    a = z[1]

    y1 = []
    for i in keys11:
        y1.append((10 ** a) * (i ** m))

    # print('Exponent power-law for eigen vs weight: ', z[0])
    plt.plot(keys11, y1, 'g.', label=z[0], mfc='none')





    # Plotting out-degree distribution and fitting corresponding least square regression line
    keys2, vals2 = plotting_helperfunc(g.out_degree)
    plt.plot(keys2, vals2, 'r.')
    # Returns coefficient of power-law line
    logkeys2 = []
    logvals2 = []
    keys21 = []

    rang =len(keys2)
    for i in range(0,rang):
        if keys2[i]>0 and vals2[i]>0:
            logkeys2.append(np.math.log10(keys2[i]))
            logvals2.append(np.math.log10(vals2[i]))
            keys21.append(keys2[i])

    # z = np.polyfit(v_u, e_u, 1)
    z = np.polyfit(logkeys2, logvals2, 1)
    # Exponent or slope of power-law
    m = z[0]
    a = z[1]

    y1 = []
    for i in keys21:
        y1.append((10 ** a) * (i ** m))

    # print('Exponent power-law for eigen vs weight: ', z[0])
    plt.plot(keys21, y1, 'r.', label=z[0], mfc='none')



    # Legend of plot for showing coefficients of each least square regression line
    legend = plt.legend(title='Slope',loc='upper right')
    legend.get_frame()
    plt.show()


    #-----------Task-1-iii--------------------------#

    # exp for power-law

    plt.xscale('log')
    plt.yscale('log')
    plt.title('Powerlaw, lognormal distributions')
    plt.xlabel('Degree')
    plt.ylabel('Fraction of nodes')

    
    logkey = []
    logval = []
    leng = len(keys)
    index = 0
    
    rang =len(keys)
    for i in range(0,rang):
        if vals[i]>0 and keys[i]>0:
            logkey.append(np.math.log10(keys[i]))
            logval.append(np.math.log10(vals[i]))
    
    
    z = np.polyfit(logkeys, logvals, 1)
    # Exponent or slope of power-law
    m = z[0]
    a = z[1]

    y1 = []
    for i in keys:
        y1.append((10 ** a) * (i ** m))

    # Degree Distribution
    plt.plot(keys,vals,'r.')
    
    # Power-law
    plt.plot(keys, y1, 'b.', label=z[0], mfc='none')







    #-----log-normal----- 
    
    loglogkey=[]
    normval = []
    logx = []
    logy = []
    loglogx = []
    rang =len(logkey)
    for i in range(0,rang):
        if logkey[i]>0 and keys[i]>0:
            loglogkey.append(np.math.log(logkey[i]))
            normval.append((logval[i]))
            logx.append(keys[i])
            loglogx.append(np.math.log10(keys[i]))
    
    
    z = np.polyfit(loglogkey, normval, 1) 
    m= z[0]
    a = z[1]
    
    y1 = []
    for i in loglogx:
        y1.append(10**(a+m*np.math.log(i)))
    
    # log-normal
    plt.plot(logx,y1,'g.',mfc = 'none')

    legend = plt.legend(title='Slope', loc='upper right')
    legend.get_frame()
    
    #-----exp----------

    """
    
    normkey=[]
    loglogval = []
    logx = []
    rang =len(logkey)
    for i in range(0,rang):
        if logval[i]>=0 and keys[i]>0:
            normkey.append((logkey[i]))
            print("line571:",logval[i])
            loglogval.append(np.math.log(logval[i]))
            logx.append((keys[i]))
    
    print("line 574:",len(normkey))
    print("line575:",len(loglogval))
    z = np.polyfit(normkey, loglogval, 1) 
    m= z[0]
    a = z[1]
    
    y1 = []
    for i in logx:
        y1.append(a*np.math.exp(m*i))
    
    #exp
    plt.plot(logx,y1,'y.',label=z[0],mfc='none')
    
    
    """

    plt.show()



def getTopPoints(v,e,ou):

    # outofnorm = dict()
    # length = len(x)
    # for i in range(0,length):
    #     if x[i] in outofnorm:
    #         if outofnorm[x[i]] < ou[i]:
    #             outofnorm[x[i]] = ou[i]
    #     else:
    #         outofnorm[x[i]] = ou[i]
    #
    # # print("source table values: ", sorted(source_table.values()))
    # # print("source table: ", sorted(outofnorm.items(), key=itemgetter(1)))
    # dictsorted = sorted(outofnorm.items(), key=itemgetter(1))
    # dictsorted = dictsorted[-20:]
    # k=[]
    # v=[]
    # for a,b in dictsorted:
    #     k.append(a)
    #     v.append(b)

    ou1 = sorted(ou)
    ou1 = ou1[-20:]
    key = []
    val = []

    for i in ou1:
        j = ou.index(i)
        key.append(v[j])
        val.append(e[j])

    return key,val


def binning():

    data = (np.random.random(10000) * 10) ** 3

    # log-scaled bins
    bins = np.logspace(0, 4, 50)
    widths = (bins[1:] - bins[:-1])

    # Calculate histogram
    hist = np.histogram(data, bins=bins)
    # normalize by bin width
    hist_norm = hist[0] / widths

    # plot it!
    plt.bar(bins[:-1], hist_norm, widths)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def lines():

    # Star
    plt.xscale('log')
    plt.yscale('log')

    k = [1,2,3,10000]
    v = [1,2,3,10000]
    p = np.polyfit(k, v, 1)
    plt.plot(k, np.polyval(p, k), 'b-', label=p[0])
    legend = plt.legend(loc='upper right')
    legend.get_frame()
    # plt.show()

    # Clique
    plt.xscale('log')
    plt.yscale('log')


    #square without one diagnol
    k = [1,2,4,10000]
    v = [2,4,8,20000]

    p = np.polyfit(k, v, 1)
    plt.plot(k, np.polyval(p, k), 'b-', label=p[0])
    legend = plt.legend(loc='upper left')
    legend.get_frame()
    plt.show()


# Task-2
# Constructing 1.5 egonetwrok
def egonetwork(g):

    c =0

    # Edges list
    e_u = []
    # Nodes list
    v_u = []
    # Weight of egonet
    w_u = []
    # Principal eigenvalue of egonet
    l_u = []

    nodes = g.nodes()

    for node in nodes:

        egonet_g = nx.Graph()
        adjlist = g.adj[node]

        neighofnode = []
        weight =0

        for i in adjlist:
            neighofnode.append(i)
            egonet_g.add_edge(node, i, weight=g[node][i]['weight'])
            weight = weight + g[node][i]['weight']


        #Adding edges for alter egos
        numof_c = len(neighofnode)
        for i in range(0, numof_c):
            for j in range(i + 1, numof_c):
                if g.has_edge(neighofnode[i], neighofnode[j]):
                    node1 = neighofnode[i]
                    node2 = neighofnode[j]
                    egonet_g.add_edge(node1, node2, weight=g[node1][node2]['weight'])
                    weight = weight + g[node1][node2]['weight']

        numedges = len(egonet_g.edges())

        if node in neighofnode:
            numnodes = len(neighofnode)

        else:
            numnodes = len(neighofnode) + 1

        # List of total number of nodes of each egonet
        v_u.append(numnodes)
        # List of total number of edges
        e_u.append(numedges)

        # Calculating principal eigen value of weighted adjacency matrix of egonetwork
        A = nx.adjacency_matrix(egonet_g)
        w, v = LA.eig(A.todense())

        # List of Principal eigen value of weighted adjacency matrix
        l_u.append(w[0])

        # Total weight of each egonet
        w_u.append(weight)



    # Plotting
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Total number of nodes in each egonetwork')
    plt.ylabel('Total number of edges for corresponding nodes in egonetwork')


    plt.plot(v_u, e_u, 'b.')


    # plt.show()



    #----------------- Task-2-(ii)----------------------#

    # Creating nd-array
    v_u1 = np.asarray(v_u)
    e_u1 = np.asarray(e_u)

    bins = np.logspace(0, 5, 50)
    # Grouping input into 50 equal width bins
    groups = np.digitize(v_u1, bins)
    y = []

    # groups now holds the index of the bin into which v_u1[i] falls
    # looping through all bin indexes and selecting the corresponding edges
    # and performing aggregation on the selected edges
    bin_median = {}
    for i in range(1, len(bins) + 1):
        # Edges corresponding to that bin
        selected_edges = e_u1[groups == i]

        # If a bin has edges
        if len(selected_edges)>0:
            # Median of edges in that bin
            bin_median[i] = np.median(selected_edges)
            y.append(bin_median[i])

        # If a bin does not have edges
        else:
            y.append(0)

    x1=[]
    y1=[]

    for i in range(0,len(y)):
        if y[i] !=0:
            x1.append(bins[i])
            y1.append(y[i])



    # plt.xscale('log')
    # plt.yscale('log')

    # plt.plot(bins, y, 'r.')
    # z = np.polyfit(bins, y, 1)
    # plt.plot(bins, np.polyval(z, bins), 'b.', label=z[0])
    plt.plot(x1, y1, 'y.',label='Median values')


    # Returns coefficient of power-law line
    logx1 = []
    logy1 = []

    rang =len(x1)
    for i in range(0,rang):
        if x1[i]>0 and y1[i]>0:
            logx1.append(np.math.log10(x1[i]))
            logy1.append(np.math.log10(y1[i]))

    # z = np.polyfit(v_u, e_u, 1)
    z = np.polyfit(logx1, logy1, 1)
    # Exponent or slope of power-law
    m = z[0]
    a = z[1]

    y1 = []
    for i in x1:
        y1.append((10**a) * (i ** m))

    # print('Exponent power-law for eigen vs weight: ', z[0])
    plt.plot(x1, y1, 'c.', label=z[0], mfc='none')



    # legend = plt.legend(title="Slope of Least square regression line:", loc='upper right')
    # legend.get_frame()
    # plt.show()


    #------Task-2-(iii)-----#
    # Star
    # plt.xscale('log')
    # plt.yscale('log')

    k = [1, 2, 3, 10000]
    v = [0.1, 0.2, 0.3, 1000]
    p = np.polyfit(k, v, 1)
    # plt.plot(k, np.polyval(p, k), 'm-', label=p[0])
    # legend = plt.legend(loc='upper right')
    # legend.get_frame()
    # plt.show()
    plt.plot(k,v,'m-',label=1)

    # Clique
    plt.xscale('log')
    plt.yscale('log')

    # square without one diagnol
    k = [1, 2, 4, 10000]
    v = [1, 4, 16, 100000000]

    # p = np.polyfit(k, v, 1)
    # plt.plot(k, np.polyval(p, k), 'g-', label=p[0])
    # legend = plt.legend(title='slope', loc='upper left')
    # legend.get_frame()
    plt.plot(k,v,'g-',label=2)
    legend = plt.legend(title="Slope:", loc='upper right')
    legend.get_frame()

    plt.show()



    #-----Task-2-(iv)-------#

    #Plotting Plower law eigen values vs weight of egonetwork

    plt.xscale('log')
    plt.yscale('log')
    plt.title('Eigenvalue versus total weight of each egonetwork')
    plt.xlabel('Total Weight of egonetowrk')
    plt.ylabel('Principal EigenValue of egonetwork')
    plt.plot(w_u, l_u, 'b.')

    # Returns coefficient of power-law line
    logl_u = []
    logw_u = []

    rang =len(l_u)
    for i in range(0,rang):
        if l_u[i]>0:
            logl_u.append(np.math.log10(l_u[i]))
            logw_u.append(np.math.log10(w_u[i]))



    # z = np.polyfit(v_u, e_u, 1)
    z = np.polyfit(logw_u, logl_u, 1)
    # Exponent or slope of power-law
    m = z[0]
    a = z[1]

    y1 = []
    for i in w_u:
        y1.append((10**a) * (i ** m))

    print('Slope of power-law for eigenvalues vs weight: ', z[0])
    plt.plot(w_u, y1, 'r.', label=z[0], mfc='none')
    legend = plt.legend(title="Slope of Least square regression line:", loc='upper right')
    legend.get_frame()
    plt.show()



    # Plotting Powerlaw for Edges versus Nodes of ego network
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Edges versus Nodes')
    plt.xlabel('Nodes in egonetwork')
    plt.ylabel('Edges in egonetwork')
    plt.plot(v_u, e_u, 'b.')

    # Returns coefficient of power-law line
    logv_u = []
    loge_u = []

    rang = len(v_u)
    for i in range(0, rang):
        if l_u[i] > 0:
            logv_u.append(np.math.log10(v_u[i]))
            loge_u.append(np.math.log10(e_u[i]))

    # for i in l_u:
    #     logv_u.append(np.math.log10(i))
    # for i in w_u:
    #     loge_u.append(np.math.log10(i))

    z = np.polyfit(logv_u, loge_u, 1)
    # Exponent or slope of power-law
    m = z[0]
    a = z[1]

    y1 = []
    for i in v_u:
        y1.append((10**a) * (i ** m))

    print('SLope of  power-law for edges vs nodes: ', z[0])
    plt.plot(v_u, y1, 'r.', label=z[0], mfc='none')
    legend = plt.legend(title="Slope of Least square regression line:", loc='upper right')
    legend.get_frame()
    plt.show()

    #--------(iii-i)----------------#



    plt.xscale('log')
    plt.yscale('log')
    plt.title('out-of-norm nodes')
    plt.xlabel('Nodes in egonetwork')
    plt.ylabel('Edges in egonetwork')
    plt.plot(v_u,e_u, 'r.')

    ou_norm = []

    # Returns coefficient of power-law line
    logv_u = []
    loge_u = []


    for i in v_u:
        logv_u.append(np.math.log10(i))
    for i in e_u:
        loge_u.append(np.math.log10(i))

    z = np.polyfit(logv_u, loge_u, 1)
    # print('exponent power-law for eigen vs weight: ', z[0])

    # Out-of-norm calculation
    m = z[0]
    c = z[1]

    rang = len(v_u)
    for point in range(0, rang):
        yu = e_u[point]
        xu = v_u[point]
        yu1 = (10**c) * pow(xu, m)

        if yu > yu1:
            ou = (yu / yu1) * np.math.log10(abs(yu - yu1) + 1)
        else:
            ou = (yu1 / yu) * np.math.log10(abs(yu1 - yu) + 1)
        ou_norm.append(ou)

    k, v = getTopPoints(v_u, e_u, ou_norm)
    plt.plot(k, v, 'gv')
    plt.show()


    #-------------3-(ii)------------------#



    plt.xscale('log')
    plt.yscale('log')
    plt.title('out-of-norm nodes')
    plt.xlabel('Weights of ego network')
    plt.ylabel('Principal Eigen value of egonetwork')
    plt.plot(w_u,l_u, 'r.')

    ou_norm = []

    # Returns coefficient of power-law line
    logl_u = []
    logw_u = []

    rang =len(l_u)
    for i in range(0,rang):
        if l_u[i]>0:
            logl_u.append(np.math.log10(l_u[i]))
            logw_u.append(np.math.log10(w_u[i]))
    # for i in l_u:
    #     logl_u.append(np.math.log10(i))
    # for i in w_u:
    #     logw_u.append(np.math.log10(i))

    z = np.polyfit(logw_u, logl_u, 1)
    # print('exponent power-law for eigen vs weight: ', z[0])

    # Out-of-norm calculation
    m = z[0]
    c = z[1]

    rang = len(w_u)
    for point in range(0, rang):
        yu = l_u[point]
        xu = w_u[point]
        yu1 = (10**c) * pow(xu, m)

        if yu > yu1:
            ou = (yu / yu1) * np.math.log10(abs(yu - yu1) + 1)
        else:
            ou = (yu1 / yu) * np.math.log10(abs(yu1 - yu) + 1)
        ou_norm.append(ou)

    k, v = getTopPoints(w_u, l_u, ou_norm)
    plt.plot(k, v, 'gv')
    plt.show()





#Task-4

#YOu have to exclude weekends
def temporalgraph_ranking():

    g = nx.DiGraph()
    gg = nx.DiGraph()


    with open("/Users/sankeerthjagini/Documents/Studies/NetSci/Project/532projectdataset.txt") as file:

        content = file.read().splitlines()
        numberofelems = len(content)

        init = content[0].split()
        inittime = int(init[0])
        index = 0

        timestamp = dt.datetime.fromtimestamp(inittime)
        timestamp1 = timestamp
        timestamp = str(timestamp)

        date_list = timestamp.split()

        date = date_list[0]

        num = numberofelems
        in_str = date
        in_time = timestamp1


        #
        # for i in range(0, num):
        #     gg.add_edge(df_source[i], df_target[i])
        # print("Nodes: ", len(gg.nodes()))
        # total = len(gg.nodes())



        exp = []
        time = []
        geo_desic = []
        time_geo = []

        ratio_list=[]
        time_component = []
        graph_signi1 = []
        graph_signi2 = []

        for i in range(0, num):

            elems = content[i].split()
            timestamp = dt.datetime.fromtimestamp(int(elems[0]))
            timestamp2 = timestamp

            timestamp = str(timestamp)
            date_list = timestamp.split()
            date = date_list[0]

            input_str = date


            if in_str == input_str:

                extra = 0
                if g.has_edge(elems[1],elems[2]):
                    extra = g[elems[1]][elems[2]]['weight']
                g.add_edge(elems[1], elems[2], weight= 1 + extra)

            else:



                # Number of unique recipients
                count = 0
                # print(g.out_degree)
                for i, j in g.out_degree:
                    count = count + j
                # print(count)

                # Number of mails
                weight = 0
                for i, j in g.edges:
                    weight = weight + g[i][j]['weight']
                # print(weight)

                graph_signi1.append(weight)
                graph_signi2.append(count)


                # time_component.append(in_str)
                time_component.append((in_time - timestamp1).days)


                in_str = input_str
                in_time = timestamp2

                g = nx.DiGraph()
                g.add_edge(elems[1], elems[2], weight=1)

        index =1
        significant_graph = []
        significant_time = []
        significant_graph1 = []
        significant_time1 = []


        for i in range(1,len(graph_signi1)):
            if graph_signi1[i]/graph_signi1[i-1]>2 and graph_signi1[i]>700:
                significant_graph.append(graph_signi1[index])
                significant_time.append(time_component[index])

            if graph_signi2[i]/graph_signi1[i-1]>2 and graph_signi1[i]>50:
                significant_graph1.append(graph_signi2[index])
                significant_time1.append(time_component[index])



            index = index+1

        plt.plot(significant_time,significant_graph,'r-')
        # plt.show()
        sort_significant_time = []
        sort_significant_time1 = []

        ou1 = sorted(significant_graph,reverse=True)

        for i in ou1:
            j = significant_graph.index(i)
            sort_significant_time.append(significant_time[j])


        ou2 =  sorted(significant_graph1,reverse=True)
        for i in ou2:
            j = significant_graph1.index(i)
            sort_significant_time1.append(significant_time1[j])

        print(sort_significant_time[:100])
        print(sort_significant_time1[:100])
        plt.plot(significant_time1, significant_graph1, 'b-')
        plt.show()


if __name__ == '__main__':


    # Task-1-(i)
    g = construct_Digraph()
    print("Number Of nodes: ", len(g.nodes))
    print("Number Of edges: ", len(g.edges))
    min_max_avg(g)

    # Task-1-(ii)
    plotting(g)

    # Task-2
    g1 = construct_graph()
    egonetwork(g1)

    temporalgraph_ranking()





