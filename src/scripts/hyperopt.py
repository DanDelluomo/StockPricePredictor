#!/usr/bin/env python

def objective123(params):
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    params = {k: v for k, v in params.items() if k not in ('batch_size', 'learning_rate')}
    trainer = Trainer(epochs=1, batch_size=batch_size)
    estimator = deepar.DeepAREstimator(
        freq="1D",
        prediction_length=prediction_length,
        trainer=trainer,
        **params
    )
    predictor = estimator.train(training_data=data)
    prediction = next(predictor.predict(data))
    accuracy = mean_squared_error(test[:prediction_length].Close, prediction.mean)
    return {'loss': accuracy, 'status': STATUS_OK}


search_space = {
    'num_layers': scope.int(hp.quniform('num_layers', 1, 8, q=1)),
    'num_cells': scope.int(hp.quniform('num_cells', 30, 100, q=1)),
    'cell_type': hp.choice('cell_type', ['lstm', 'gru']),
    'batch_size': scope.int(hp.quniform('batch_size', 16, 256, q=1)),
    'learning_rate': hp.quniform('learning_rate', 1e-5, 1e-1, 0.00005),
    'context_length': scope.int(hp.quniform('context_length', 1, 200, q=1)),
}

trials = Trials()
best = fmin(
    objective123,
    space=search_space,
    algo=tpe.suggest,
    max_evals=10,
    trials=trials,
)