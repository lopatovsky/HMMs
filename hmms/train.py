import numpy
import hmms

def multi_train_ct(  hidden_states, times, data, runs, iteration = 10, **kwargs ):
        """Run multiple Baum-welch algorithms, always with different random initialization.
           kwargs:
                method: 'exp' - [default] Use exponential distribution for random initialization
                        'unif' - Use uniform distribution for random initialization

                ret: 'all' - Return all trained models, sorted by their probability estimation
                        'best' - [default] Return only the model with the best probability estimation
        """

        if 'method' not in kwargs : kwargs['method'] = 'exp'   #default exponential
        if 'ret' not in kwargs : kwargs['ret'] = 'best'  #default best

        models = []
        outputs = numpy.max( data ) + 1

        #print(hidden_states, outputs)

        for i in range( runs ):
            model = hmms.CtHMM.random( hidden_states, outputs, method = kwargs['method']  )
            graph = model.baum_welch( times, data, iteration, est = True)
            models.append( (model, graph)  )


        models.sort(key=lambda x: x[1][-1] , reverse=True)

        if kwargs['ret'] == 'all': return models

        return models[0]

def multi_train_dt(  hidden_states, data, runs, iteration = 10, **kwargs ):
        """Run multiple Baum-welch algorithms, always with different random initialization.
           kwargs:
                method: 'exp' - [default] Use exponential distribution for random initialization
                        'unif' - Use uniform distribution for random initialization

                ret: 'all' - Return all trained models, sorted by their probability estimation
                        'best' - [default] Return only the model with the best probability estimation
        """

        if 'ret' not in kwargs : kwargs['ret'] = 'best'  #default best

        models = []
        outputs = numpy.max( data ) + 1

        print(hidden_states, outputs)

        for i in range( runs ):
            model = hmms.DtHMM.random( hidden_states, outputs )
            graph = model.baum_welch( data, iteration, est = True)
            models.append( (model, graph)  )


        models.sort(key=lambda x: x[1][-1] , reverse=True)

        if kwargs['ret'] == 'all': return models

        return models[0]

