from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer

def build_metadata(df):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    return metadata

def train_gaussian_copula(df):
    metadata = build_metadata(df)
    synthesizer = GaussianCopulaSynthesizer(metadata=metadata)
    synthesizer.fit(df)
    return synthesizer

def train_ctgan(df):
    metadata = build_metadata(df)
    synthesizer = CTGANSynthesizer(metadata=metadata, epochs=300, verbose=True)
    synthesizer.fit(df)
    return synthesizer

def train_tvae(df):
    metadata = build_metadata(df)
    synthesizer = TVAESynthesizer(metadata=metadata, epochs=300)
    synthesizer.fit(df)
    return synthesizer

def generate_synthetic_data(synthesizer, num_rows):
    synthetic_df = synthesizer.sample(num_rows=num_rows)
    return synthetic_df