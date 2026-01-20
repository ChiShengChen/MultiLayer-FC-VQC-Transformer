# main_bs.py
from portfolio import *
from functions import ModelRunner


params = {
    'T': 1,
    'r': 0.1,
    'num_steps': 10,
    'num_simulations': 1000,
    'hidden_size': 64,
    'num_layers': 3,
    'vqc_depth': 3,
    'batch_size': 64,
    'learning_rate': 1e-2,
    'epochs': 100,
    'TrainingModel': [
                    #   'DNN',
                    #   'QNN_Q3',
                    #   'QNN_Q3_Parallel' 
                    #   'QNN_QSquared'     
                      'XGBoost',   
                      'CatBoost'
                      ], 

    'data': BS_data_36, #choose BS_data_1, BS_data_10, BS_data_100
    'portfolio': BS_portfolio_36, # choose BS_portoflio_1, BS_portoflio_10, BS_portoflio_100 
    'portfolio_name': 'BS_portfolio_36', # Note! need to manual write the portfolio
    'device': 'cpu', #cpu or cuda
}

TRAIN = True # True for training a model, False for loading a saved model.TS
if TRAIN:
    runner = ModelRunner(params, train_mode=True)  # for training
    results, directory = runner.run()    
    runner.plot_relative_mae()
    runner.save_summary_to_csv()
    runner.save_and_plot_costs()
    runner.save_full_predictions()
    runner.save_history_csv()
    runner.plot_gradient_history()


# else:
#     directory_name = "RegressionComparison_BS_portfolio_100_20260116_174553" #"YOUR_SAVED_FOLDER_NAME"
#     runner = ModelRunner(params, train_mode=False, directory_name=directory_name)  # for loading
#     results, directory = runner.run()
#     runner.plot_relative_mae()
#     runner.save_summary_to_csv()
#     runner.save_full_predictions()

# for i in range(1,10,2):
#     for j in range(1,10,2):
#         params['vqc_depth'] = i
#         params['num_layers'] = j
#         params['portfolio_name'] = f'BS_portfolio_D36_L{j}_K{i}'
#         runner = ModelRunner(params, train_mode=True)  # for training
#         results, directory = runner.run()    
#         runner.plot_relative_mae()
#         runner.save_summary_to_csv()
#         runner.save_and_plot_costs()
#         runner.save_full_predictions() 


