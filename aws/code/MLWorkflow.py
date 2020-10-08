from abc import abstractmethod
import functools
import time

class MLWorkflow ( ):
    def __init__ ( self ):
        return None

    def create_workflow( self, hpo_config ):        
        if hpo_config.compute_type == 'single-CPU':
            from workflows.MLWorkflowSingleCPU import MLWorkflowSingleCPU
            return MLWorkflowSingleCPU( hpo_config )

        if hpo_config.compute_type == 'multi-CPU':
            from workflows.MLWorkflowMultiCPU import MLWorkflowMultiCPU
            return MLWorkflowMultiCPU( hpo_config )

        if hpo_config.compute_type == 'single-GPU':
            from workflows.MLWorkflowSingleGPU import MLWorkflowSingleGPU
            return MLWorkflowSingleGPU( hpo_config )

        if hpo_config.compute_type == 'multi-GPU':
            from workflows.MLWorkflowMultiGPU import MLWorkflowMultiGPU
            return MLWorkflowMultiGPU( hpo_config )
    
    @abstractmethod
    def ingest_data ( self ): pass

    @abstractmethod
    def handle_missing_data ( self, dataset ): pass

    @abstractmethod
    def split_dataset ( self, dataset, i_fold ): pass

    @abstractmethod
    def fit ( self, X_train, y_train ): pass

    @abstractmethod
    def predict ( self, trained_model, X_test ): pass

    @abstractmethod
    def score ( self, y_test, predictions ): pass

    @abstractmethod
    def save_trained_model ( self, score, trained_model ): pass

    @abstractmethod
    def cleanup ( self, i_fold ): pass
    
    @abstractmethod
    def emit_final_score ( self ): pass


def timer_decorator ( target_function ):

    @functools.wraps ( target_function )
    def timed_execution_wrapper ( *args, **kwargs ):
        start_time = time.perf_counter()
        result = target_function ( *args, **kwargs )
        print(f" --- {target_function.__name__} completed in {round( time.perf_counter() - start_time, 5 )} s")
        return result
        
    return timed_execution_wrapper