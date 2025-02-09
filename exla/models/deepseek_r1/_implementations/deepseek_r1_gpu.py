from ._base import Deepseek_R1_Base
from vllm import LLM, SamplingParams

class Deepseek_R1_GPU(Deepseek_R1_Base):
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        """
        Initializes a DeepSeek model using vLLM for efficient GPU inference.
        """
        super().__init__(model_name)
        self.model = LLM(model=model_name)
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95)

    def train(self):
        raise NotImplementedError("Training is not supported for this model")

    def inference(self, prompts, max_tokens=128):
        """
        Run inference on the model using vLLM.
        
        Args:
            prompts (str or list): Input text prompt(s) 
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            list: Generated text responses
        """
        if isinstance(prompts, str):
            prompts = [prompts]
            
        outputs = self.model.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]