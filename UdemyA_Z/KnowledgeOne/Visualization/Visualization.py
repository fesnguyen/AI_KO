import matplotlib.pyplot as plt
import math

def visualize(df, kind, ncols=5, height=6, vert=True, tight=True):
  """
  :param df: Data frame
  :param kind: Kind of plot
  :param ncols: num of columns
  :param height: height of each plot
  :param vert: display verticaly
  :param tight: tight layout
  :return: code reveal
  """

  ncharts = len(df.columns)
  nrows = math.ceil(ncharts / ncols)
  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, height))
  if ncharts % ncols > 0:
    [axes[nrows-1, col].axis('off') for col in range((ncharts % ncols), ncols)]

  for i, col_name in enumerate(df.columns):
    col = i % ncols
    row = i // ncols
    df[col_name].plot(kind=kind, vert=vert, ax=axes[row, col])

  if tight:
      plt.tight_layout()

  plt.show()

  return r"""
    ncharts = len(df.columns)
    nrows = math.ceil(ncharts / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, height))
    if ncharts % ncols > 0:
      [axes[nrows-1, col].axis('off') for col in range((ncharts % ncols), ncols)]
  
    for i, col_name in enumerate(df.columns):
      col = i % ncols
      row = i // ncols
      df[col_name].plot(kind=kind, vert=vert, ax=axes[row, col])
  
    if tight:
        plt.tight_layout()
  
    plt.show()
    
    return
  """