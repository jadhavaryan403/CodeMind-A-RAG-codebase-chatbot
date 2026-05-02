from langchain_core.callbacks import BaseCallbackHandler


class TokenTrackingCallback(BaseCallbackHandler):
    '''Custom callback handler to track token usage during LLM interactions.'''
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_llm_end(self, response, **kwargs):
        '''Called when the LLM finishes generating a response. Extracts token usage from the response and updates counters.'''
        try:
            usage = response.llm_output.get("token_usage", {})

            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
        except Exception:
            pass