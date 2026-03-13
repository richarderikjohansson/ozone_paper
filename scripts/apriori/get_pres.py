import pandas as pd

url = "https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions"
df = pd.read_html(url, header=0)[0]

# Save all columns to separate .txt files
for col in df.columns:
    df[col].to_csv(f"{col}.txt", index=False, header=True)
