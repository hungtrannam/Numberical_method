import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as plt_gs
sns.set_theme(context = 'notebook', color_codes = True)

# Load dataset in seaborn
ans = sns.load_dataset("anscombe")
data = ans.loc[ans.dataset == "I"]


# Calculating regression coefficients
def estimate_coef(x, y):
  # mean of x and y vector
  m_x = np.nanmean(x)
  m_x2 = np.nanmean(x*x)
  m_y = np.nanmean(y)
  m_xy = np.nanmean(x*y)
  # calculating regression coefficients
  M = np.matrix([[m_x, 1], [m_x2, m_x]])
  b = np.array([m_y, m_xy])
  X = np.linalg.solve(M,b)
  return X
def estimate_poly_coef(x, y):
  # mean of x and y vector
  m_x = np.nanmean(x)
  m_y = np.nanmean(y)
  m_x2 = np.nanmean(x*x)
  m_x3 = np.nanmean(x*x*x)
  m_x4 = np.nanmean(x*x*x*x)
  m_xy = np.nanmean(x*y)
  m_x2y = np.nanmean(x*x*y)
  # calculating regression coefficients
  M = np.matrix([[m_x2, m_x, 1], [m_x3, m_x2, m_x], [m_x4, m_x3,
  m_x2]])
  b = np.array([m_y, m_xy, m_x2y])
  X = np.linalg.solve(M,b)
  return X
def estimate_sincos(x, y):
  # mean of x and y vector
  m_sx = np.nanmean(np.cos(x))
  m_cx = np.nanmean(np.sin(x))
  m_sx2 = np.nanmean(np.sin(x)*np.sin(x))
  m_cx2 = np.nanmean(np.cos(x)*np.cos(x))
  m_scx = np.nanmean(np.sin(x)*np.cos(x))
  m_y = np.nanmean(y)
  m_ys = np.nanmean(y*np.sin(x))
  m_yc = np.nanmean(y*np.cos(x))
  # calculating regression coefficients
  M = np.matrix([[m_cx2, m_scx, m_cx], [m_scx, m_sx2, m_sx], [m_cx, m_sx,1]])
  b = np.array([m_yc, m_ys, m_y])
  X = np.linalg.solve(M,b)
  return X

def error(x,y,b):
  # estimating error for linear regression
  e2 = np.power((b[0]*x + b[1]) - y, 2)
  E = np.sum(e2)
  return (E)
def error(x,y,b):
  # estimating error for polynomial regression
  e2 = np.power(b[0]*x*x + b[1]*x + b[2] - y, 2)
  E = np.sum(e2)
  return (E)
def error(x,y,b):
  # estimating error for nonlinear regression ex^bx
  e2 = np.power(np.exp(b[1])*x*np.exp(b[0]*x) - y, 2)
  E = np.sqrt(np.sum(e2))
  return (E)
def error(x,y,b):
  # estimating error for nonlinear regression acosx + bsinx +c
  e2 = np.power((b[0]*np.cos(x) + b[1]*np.sin(x) + b[2]) - y, 2)
  E = np.sqrt(np.sum(e2))
  return (E)

def main():
  # estimating coefficients
  c = estimate_coef(data.x, data.y)
  err = error(data.x, data.y, c)
  print("coefficients:\na = {} \nb = {}".format(c[0], c[1]))
  print("Error: {}".format(err))
  # plotting regression line
  plot_regression_line(data.x, data.y, c)
if __name__ == "__main__":
  main()
