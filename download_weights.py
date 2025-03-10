import os
import gdown

file_ids = {
    "FARE": "1IMtb5SG1ajYphR8cK-3w3Nr7zvAHa8bi",
    "TeCoA": "1m4Iw9pCjtBHj7OVqHFlRu2OkO0rd7C7j",
    "PMG_AFT": "1JMXdMheNaYWiwqcWI0tvRrp6MBweQagn",
}

os.makedirs("./AFT_model_weights", exist_ok=True)

for name, fid in file_ids.items():
    url = f"https://drive.google.com/uc?id={fid}"
    output = f"./AFT_model_weights/{name}.pth.tar"
    gdown.download(url, output, quiet=False)
    print(f"Downloaded {name} weights to {output}")