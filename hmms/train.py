import numpy

def multi( self, model, times, data, runs, iteration = 10, **kwargs ):
        """Run multiple Baum-welch algorithms, always with different random initialization"""
        
        graph = numpy.empty( (runs,iteration+1) )

        for i in range( runs ):
            print(i)
            graph[i] = self.baum_welch_graph( times, data, iteration )

        return graph
