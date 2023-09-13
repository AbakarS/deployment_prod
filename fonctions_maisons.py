import pandas as pd

def extraire_la_première_lettre(serie):

  return pd.DataFrame(serie.str[0])
