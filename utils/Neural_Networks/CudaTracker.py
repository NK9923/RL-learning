import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

class CUDAMemoryTracker:
    def __init__(self, agent, output_dir, device_id=0):
        self.DNN_agent = agent
        self.device_id = device_id
        self.device = None
        self.tracking_enabled = False  
        self.allocated_memory = 0
        self.reserved_memory = 0
        self.general_allocated_memory = []
        self.model_memory = []
        self.data_memory = []

        self.target_dir = os.path.join(output_dir, "GPU_usage_info")  
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)              

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
            self.tracking_enabled = True  # Enable tracking if CUDA is available

    def track_memory_usage(self) -> None:
        if not self.tracking_enabled:
            return  # Tracking is disabled if CUDA is not available

        allocated_memory = torch.cuda.memory_allocated(self.device)
        self.general_allocated_memory.append(allocated_memory)

        reserved_memory = torch.cuda.memory_reserved(self.device)

        model_memory = self._get_model_memory_usage()
        data_memory = allocated_memory - model_memory
        self.data_memory.append(data_memory)
        self.model_memory.append(model_memory)
        
    def _get_model_memory_usage(self) -> None:
        model_memory = sum(param.nelement() * param.element_size() for param in self.DNN_agent.policy_net.parameters()) 
        return model_memory

    def plotGPUconsumption(self) -> None:
        
        def GB_converter(x):
            return round(x / (1024 ** 3), 2)
      
        model_memory = pd.Series(self.model_memory)
        model_memory = model_memory.cumsum()  
        model_memory = model_memory.apply(GB_converter)
        
        data_memory = pd.Series(self.data_memory)
        data_memory = data_memory.cumsum()  
        data_memory = data_memory.apply(GB_converter)        

        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111, facecolor='lightgray')
        ax.grid(True, linestyle='--', color='gray')
        plt.plot(model_memory, label='Model Memory', color='deeppink')
        plt.plot(data_memory, label=f'Data Memory', color='green')

        plt.xlabel('Iterations')
        plt.ylabel('Memory Usage (GB)')
        plt.title('Memory Usage Chart')
        plt.legend()
        
        plt.title("CUDA Memory Usage Chart")   
        
        plt_name = "GPU_usage.pdf"     
        plt.savefig(os.path.join(self.target_dir, plt_name), format='pdf')           

    def __repr__(self) -> str:
        if not self.tracking_enabled:
            return "CUDA is not available, tracking is disabled."

        self.plotGPUconsumption()        

        cuda_output = f"""
            ---------------------------------------------------------------------
            |{'Device':<30}|{torch.cuda.get_device_name(self.device):<36}|
            |{'Total model memory:':<30}|{round(sum(self.model_memory) / (1024 ** 3), 1):<33} GB|
            |{'Total memory allocated:':<30}|{round(sum(self.general_allocated_memory) / (1024 ** 3), 1):<33} GB|
            |{'Total data memory allocated:':<30}|{round(sum(self.data_memory) / (1024 ** 3), 1):<33} GB|
            ---------------------------------------------------------------------
            """
        return cuda_output
             
