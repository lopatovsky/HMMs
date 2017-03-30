import numpy

def multi_( self, runs, times, data, iteration = 10 ):
        """Return all convergences in the array"""
        graph = numpy.empty( (runs,iteration+1) )

        for i in range( runs ):
            print(i)
            graph[i] = self.baum_welch_graph( times, data, iteration )

        return graph
