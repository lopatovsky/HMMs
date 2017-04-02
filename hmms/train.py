import numpy
import hmms

def multi_train(  hidden_states, times, data, runs, iteration = 10, **kwargs ):
        """Run multiple Baum-welch algorithms, always with different random initialization.
           kwargs:
                method: 'exp' - [default] Use exponential distribution for random initialization
                        'unif' - Use uniform distribution for random initialization

                return: 'all' - Return all trained models, sorted by their probability estimation
                        'best' - [default] Return only the model with the best probability estimation
        """

        if 'method' not in kwargs : kwargs['method'] = 'exp'

        models = []
        outputs = numpy.max( data ) + 1

        print(hidden_states, outputs)

        for i in range( runs ):
            model = hmms.CtHMM.random( hidden_states, outputs, method = kwargs['method']  )  #todo zafunguje ak nema kwarg method vobec?
            graph = model.baum_welch( times, data, iteration, est = True)
            models.append( ( model, graph )  )

        return models
