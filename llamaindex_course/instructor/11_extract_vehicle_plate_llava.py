import requests
import base64
from PIL import Image
import io
import webp  # WebP support
import json
from rich import print
from pydantic import BaseModel


class Vehicle(BaseModel):
    """A vehicle description"""

    plate: str = None
    color: str = None
    constructor: str = None
    model: str = None


def download_image_from_url_and_encode_to_base64(url: str) -> str:
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()  # Check if the request was successful
        image_content = response.content

        # Convert to PNG
        with Image.open(io.BytesIO(image_content)) as image:
            if image.format == "WEBP":
                # Convert WebP to PNG
                webp_data = image_content
                image_data = webp.WebPData(webp_data)
                image_data.to_png().save("temp.png")
                with open("temp.png", "rb") as f:
                    image_content = f.read()
                os.remove("temp.png")  # Remove the temporary file
            elif image.format == "JPEG":
                # Convert JPEG to PNG
                with io.BytesIO() as output:
                    image.save(output, format="PNG")
                    image_content = output.getvalue()

            with io.BytesIO() as output:
                output.write(image_content)
                image_content = output.getvalue()

        base64_encoded_image = base64.b64encode(image_content).decode("utf-8")
        return base64_encoded_image
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return ""


def extract(url: str) -> Vehicle:

    md_json_result = call_llava(
        """
                             
                             Describe the vehicule, including color, plaque, constructor, model
                             
                             format as json: 
                             
                             example:
                                {
                                    "plate": "1234ABC",
                                    "color": "red",
                                    "constructor": "Toyota",
                                    "model": "Corolla"
                                }
                                
                                Format the JSON as markdown in the response.
                             
                             """,
        url,
    )

    print("lava result: ", md_json_result)
    # extract the json from the markdown
    json_result = md_json_result.split("```json")[1].split("```")[0]
    json_data = json.loads(json_result)
    return Vehicle.model_validate(json_data)


def call_llava(prompt: str, url: str) -> str:
    """
    Call the llava model to extract the vehicle plate from an image
    """
    ollama_api_url = "http://localhost:11434/api/generate"
    # Download the image and encode it to base64
    print(f"Downloading image from {url}")
    image_base64 = download_image_from_url_and_encode_to_base64(url)
    print(f"Image downloaded and encoded to base64: {image_base64:.100}...")
    data = {
        "model": "llava",
        "prompt": prompt,
        "images": [image_base64],
        "max_tokens": 32000,
        "stream": False,
    }
    print(f"Calling Ollama API with data: {data}")
    response = requests.post(ollama_api_url, json=data, timeout=120)
    result = response.json()
    description = result["response"]
    return description


TEST_URL = "https://images.bfmtv.com/OzP7CQB9bPlD-BxuwG2tzjZn-Nc=/6x4:1254x706/1248x0/images/-158334.jpg"

example = extract(TEST_URL)

print(example)
