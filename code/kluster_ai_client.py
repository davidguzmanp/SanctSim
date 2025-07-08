import requests
import backoff
from requests.exceptions import HTTPError, RequestException

class KlusterAIError(Exception):
    """Custom exception for KlusterAI errors."""
    pass

class RateLimitError(KlusterAIError):
    """Exception raised when rate limit is exceeded."""
    pass

class KlusterAIClient:
    def __init__(self, api_key, base_url="https://api.kluster.ai/v1", api_version='2024-12-01-preview', model_name='None'):
        """
        Initialize the KlusterAIClient with the provided API key and base URL.

        Parameters:
        - api_key (str): Your KlusterAI API key.
        - base_url (str): The base URL for KlusterAI API. Default is 'https://api.kluster.ai/v1'.
        - api_version (str): The API version to use. Default is '2024-12-01-preview'.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.api_version = api_version
        self.deployment_name = model_name
        self.total_cost = 0  # Track total cost
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    @backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
    def send_request(self, model_name, prompt, max_tokens=9000, temperature=1.0, top_p=1.0, **kwargs):
        """
        Send a prompt to the specified model via the KlusterAI API.

        Parameters:
        - model_name (str): The name of the model to use.
        - prompt (str): The prompt to send to the model.
        - max_tokens (int): The maximum number of tokens to generate.
        - temperature (float): Sampling temperature.
        - top_p (float): Nucleus sampling parameter.
        - kwargs: Additional parameters for the API.

        Returns:
        - str: The generated response from the model.
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract generated text
            generated_text = data['choices'][0]['message']['content'].strip()

            # Token usage
            prompt_tokens = data['usage']['prompt_tokens']
            completion_tokens = data['usage']['completion_tokens']

            # Calculate and accumulate cost
            cost = self.calculate_price(model_name, prompt_tokens, completion_tokens)
            self.total_cost += cost

            return generated_text

        except HTTPError as http_err:
            if response.status_code == 429:
                raise RateLimitError("Rate limit exceeded. Retrying...") from http_err
            else:
                raise KlusterAIError(f"HTTP error occurred: {http_err}") from http_err
        except RequestException as req_err:
            raise KlusterAIError(f"Request error occurred: {req_err}") from req_err
        except KeyError as key_err:
            raise KlusterAIError(f"Unexpected response structure: {key_err}") from key_err

    def calculate_price(self, model_name, prompt_tokens, completion_tokens):
        """
        Calculate the price based on the token usage and model pricing.

        Parameters:
        - model_name (str): The name of the model used.
        - prompt_tokens (int): Number of tokens in the prompt.
        - completion_tokens (int): Number of tokens in the completion.

        Returns:
        - float: Approximate cost in USD.
        """
        # Model pricing per 1,000 tokens
        model_pricing = {
            'klusterai/Meta-Llama-3.1-8B-Instruct-Turbo': {
                'input': 0.005,   # $0.005 per 1,000 tokens
                'output': 0.015    # $0.015 per 1,000 tokens
            },
            'klusterai/Meta-Llama-3.1-405B-Instruct-Turbo': {
                'input': 0.03,
                'output': 0.09
            },
            'klusterai/Meta-Llama-3.3-70B-Instruct-Turbo': {
                'input': 0.02,
                'output': 0.06
            },
            'deepseek-ai/DeepSeek-R1': {
                'input': 0.01,
                'output': 0.03
            },
            # Add more models and their pricing as needed
        }

        pricing = model_pricing.get(model_name, None)
        if not pricing:
            # Default pricing if model not found
            input_price = 0.0
            output_price = 0.0
        else:
            input_price = pricing['input']
            output_price = pricing['output']

        input_cost = (prompt_tokens / 1000) * input_price
        output_cost = (completion_tokens / 1000) * output_price
        total_cost = input_cost + output_cost

        return total_cost

    def get_total_cost(self):
        """
        Get the total accumulated cost of all API calls.

        Returns:
        - float: Total cost in USD
        """
        return self.total_cost

    # ------------------------ Batch Operations ------------------------

    def submit_batch_job(self, input_file_id, endpoint="/v1/chat/completions", completion_window="24h", metadata=None):
        """
        Submit a Batch job.

        Parameters:
        - input_file_id (str): The ID of the uploaded input file.
        - endpoint (str): The API endpoint to use. Default is '/v1/chat/completions'.
        - completion_window (str): Time window for completion. Options: '1h', '3h', '6h', '12h', '24h'.
        - metadata (dict): Optional custom metadata.

        Returns:
        - dict: The created Batch object.
        """
        url = f"{self.base_url}/batches"
        payload = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
            "metadata": metadata or {}
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except HTTPError as http_err:
            raise KlusterAIError(f"HTTP error occurred: {http_err}") from http_err
        except RequestException as req_err:
            raise KlusterAIError(f"Request error occurred: {req_err}") from req_err

    def retrieve_batch_job(self, batch_id):
        """
        Retrieve a Batch job by ID.

        Parameters:
        - batch_id (str): The ID of the Batch to retrieve.

        Returns:
        - dict: The Batch object.
        """
        url = f"{self.base_url}/batches/{batch_id}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except HTTPError as http_err:
            raise KlusterAIError(f"HTTP error occurred: {http_err}") from http_err
        except RequestException as req_err:
            raise KlusterAIError(f"Request error occurred: {req_err}") from req_err

    def cancel_batch_job(self, batch_id):
        """
        Cancel a Batch job by ID.

        Parameters:
        - batch_id (str): The ID of the Batch to cancel.

        Returns:
        - dict: The updated Batch object with status 'cancelling'.
        """
        url = f"{self.base_url}/batches/{batch_id}/cancel"
        try:
            response = requests.post(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except HTTPError as http_err:
            raise KlusterAIError(f"HTTP error occurred: {http_err}") from http_err
        except RequestException as req_err:
            raise KlusterAIError(f"Request error occurred: {req_err}") from req_err

    def list_batch_jobs(self, limit=20, after=None):
        """
        List all Batch jobs with optional pagination.

        Parameters:
        - limit (int): Number of Batch jobs to retrieve (1-100). Default is 20.
        - after (str): Cursor for pagination.

        Returns:
        - dict: A list of Batch objects.
        """
        url = f"{self.base_url}/batches"
        params = {"limit": limit}
        if after:
            params["after"] = after
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except HTTPError as http_err:
            raise KlusterAIError(f"HTTP error occurred: {http_err}") from http_err
        except RequestException as req_err:
            raise KlusterAIError(f"Request error occurred: {req_err}") from req_err

    # ------------------------ File Management ------------------------

    def upload_file(self, file_path, purpose="batch"):
        """
        Upload a file to KlusterAI.

        Parameters:
        - file_path (str): Path to the file to upload.
        - purpose (str): Purpose of the file. Default is 'batch'.

        Returns:
        - dict: The uploaded File object.
        """
        url = f"{self.base_url}/files/"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
            # Note: 'Content-Type' is set automatically by 'requests' when using 'files' parameter
        }

        files = {
            "file": open(file_path, "rb")
        }
        data = {
            "purpose": purpose
        }

        try:
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            return response.json()
        except HTTPError as http_err:
            raise KlusterAIError(f"HTTP error occurred: {http_err}") from http_err
        except RequestException as req_err:
            raise KlusterAIError(f"Request error occurred: {req_err}") from req_err
        finally:
            files["file"].close()

    def retrieve_file_content(self, file_id):
        """
        Retrieve the content of a file by ID.

        Parameters:
        - file_id (str): The ID of the file to retrieve.

        Returns:
        - bytes: The content of the file.
        """
        url = f"{self.base_url}/files/{file_id}/content"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.content
        except HTTPError as http_err:
            raise KlusterAIError(f"HTTP error occurred: {http_err}") from http_err
        except RequestException as req_err:
            raise KlusterAIError(f"Request error occurred: {req_err}") from req_err

    # ------------------------ Model Management ------------------------

    def list_models(self):
        """
        List all supported models.

        Returns:
        - dict: A list of Model objects.
        """
        url = f"{self.base_url}/models"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except HTTPError as http_err:
            raise KlusterAIError(f"HTTP error occurred: {http_err}") from http_err
        except RequestException as req_err:
            raise KlusterAIError(f"Request error occurred: {req_err}") from req_err

  