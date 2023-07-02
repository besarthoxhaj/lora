# %%
import data
# %%
ds = data.TrainDataset()
# %%
df = ds.ds.to_pandas()
# %%
df["input_ids"].size
# %%
df["input_ids"].apply(len).max()
# %%
